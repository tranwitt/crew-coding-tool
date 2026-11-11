import ast
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from crewai.tools import tool
from config import Config, EXCLUDED
from undo import record_edit


def make_tools(cfg: Config, *, allow_tests: bool = False, on_event=None):
    def emit(action, **details):
        if on_event is not None:
            try:
                on_event({'action': action, **details})
            except Exception:
                pass  # Telemetry must never interfere with file safety.

    @tool('list_project_files')
    def list_project_files() -> str:
        """List Python files in the repository, including read-only paths."""
        files = []
        for directory, dirs, names in os.walk(cfg.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                if name.endswith('.py'):
                    relative = (Path(directory) / name).relative_to(cfg.root)
                    try:
                        cfg.file(str(relative))
                        files.append(str(relative))
                    except (ValueError, FileNotFoundError):
                        pass
            if len(files) >= 1000:
                break
        return '\n'.join(files[:1000]) or '(no editable files found)'

    @tool('read_project_file')
    def read_project_file(filename: str) -> str:
        """Read a Python source file by its project-relative filename, up to 100000 characters."""
        try:
            path = cfg.file(filename)
            if path.stat().st_size > 300_000:
                return 'ERROR: File too large to read safely'
            content = path.read_text(encoding='utf-8')
            if len(content) > 100_000:
                return 'ERROR: File exceeds 100000 characters'
            return content
        except (ValueError, OSError, UnicodeError) as exc:
            return f'ERROR: {exc}'

    @tool('patch_project_file')
    def patch_project_file(filename: str, old_text: str, new_text: str) -> str:
        """Replace one unique exact fragment in an existing Python file; syntax-check, back up, and write atomically."""
        try:
            emit('patch_attempt', file=filename)
            path = cfg.file(filename, for_edit=True)
            if not old_text:
                return 'ERROR: old_text must not be empty'
            original_bytes = path.read_bytes()
            if len(original_bytes) > 300_000:
                return 'ERROR: File too large'
            original = original_bytes.decode('utf-8')
            count = original.count(old_text)
            if count != 1:
                return f'ERROR: Expected exactly one match; found {count}'
            revised = original.replace(old_text, new_text, 1)
            if revised == original:
                emit('patch_no_change', file=filename)
                return 'ERROR: Patch would not change the file'
            if path.suffix.lower() == ".py":
                ast.parse(revised, filename=str(path))
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
            backup = cfg.backup_root / run_id / path.relative_to(cfg.root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            # Preserve permissions, and replace atomically on the same filesystem.
            fd, temp_name = tempfile.mkstemp(prefix='.agent-', suffix=path.suffix or '.txt', dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(revised.encode('utf-8'))
                    output.flush()
                    os.fsync(output.fileno())
                shutil.copymode(path, temp_name)
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            try:
                record_edit(cfg, filename, backup, original_bytes, revised.encode('utf-8'))
            except Exception:
                # Never leave an untracked edit when journaling fails.
                os.replace(backup, path)
                raise
            emit('patch_success', file=filename, backup=str(backup))
            diff = ''.join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True), fromfile=filename+' (before)', tofile=filename+' (after)'))
            return json.dumps({'status': 'updated', 'file': filename, 'backup': str(backup), 'diff': diff[:16000]})
        except (ValueError, OSError, UnicodeError, SyntaxError, PermissionError) as exc:
            emit('patch_error', file=filename, error=f'{type(exc).__name__}: {exc}')
            return f'ERROR: {type(exc).__name__}: {exc}'

    @tool('preview_project_patch')
    def preview_project_patch(filename: str, old_text: str, new_text: str) -> str:
        """Preview a targeted exact-text patch without modifying files."""
        try:
            emit('preview_attempt', file=filename)
            path = cfg.file(filename, for_edit=True)
            original = path.read_text(encoding='utf-8')
            count = original.count(old_text) if old_text else 0
            if count != 1:
                return f'ERROR: Expected one nonempty match; found {count}'
            revised = original.replace(old_text, new_text, 1)
            if path.suffix.lower() == ".py":
                ast.parse(revised, filename=str(path))
            return ''.join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True), fromfile=filename, tofile=filename+' (proposed)'))[:16000]
        except (ValueError, OSError, UnicodeError, SyntaxError, PermissionError) as exc:
            emit('preview_error', file=filename, error=str(exc))
            return f'ERROR: {exc}'

    @tool('run_pytest')
    def run_pytest(test_path: str = 'tests') -> str:
        """Run pytest against tests/ or a project-relative test_*.py file, with a 120-second timeout."""
        if not allow_tests:
            return 'ERROR: Test execution disabled. Start CLI with --allow-tests in a trusted sandbox.'
        if test_path == 'tests':
            target = cfg.root / 'tests'
        else:
            try:
                target = cfg.file(test_path)
            except (ValueError, OSError) as exc:
                return f'ERROR: {exc}'
            if not target.name.startswith('test_') or 'tests' not in target.relative_to(cfg.root).parts:
                return 'ERROR: Only test_*.py files inside tests/ can be executed'
        if not target.exists():
            return f'ERROR: Test target does not exist: {test_path}'
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(target)], cwd=cfg.root, capture_output=True, text=True, timeout=120, check=False)
            return f'Exit code: {result.returncode}\n{(result.stdout + result.stderr)[-16000:]}'
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f'ERROR: {exc}'

    return [list_project_files, read_project_file, preview_project_patch, patch_project_file, run_pytest]

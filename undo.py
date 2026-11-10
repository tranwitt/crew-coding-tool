"""Local, Git-free edit journal and safe undo for CrewAI patches."""
import hashlib
import json
import os
import tempfile
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_RUN = ContextVar('active_coding_run', default=None)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def journal_dir(cfg) -> Path:
    directory = cfg.backup_root / 'undo-journals'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def journal_path(cfg, run_id: str) -> Path:
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):
        raise ValueError('Invalid run identifier')
    return journal_dir(cfg) / (run_id + '.json')


def atomic_write(path: Path, data: bytes, mode: int | None = None):
    fd, temp = tempfile.mkstemp(prefix='.agent-undo-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if mode is not None:
            os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def record_edit(cfg, filename: str, backup: Path, before: bytes, after: bytes):
    run_id = ACTIVE_RUN.get()
    if run_id is None:
        return  # Only CLI-driven runs are journaled.
    path = journal_path(cfg, run_id)
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
    else:
        data = {'run_id': run_id, 'repo': str(cfg.root), 'created_utc': datetime.now(timezone.utc).isoformat(), 'undone': False, 'edits': []}
    data['edits'].append({'file': filename, 'backup': str(backup), 'before_sha256': digest(before), 'after_sha256': digest(after)})
    atomic_write(path, json.dumps(data, indent=2).encode('utf-8'))


def get_runs(cfg):
    results = []
    for path in journal_dir(cfg).glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('repo') == str(cfg.root) and data.get('edits'):
                results.append(data)
        except (OSError, ValueError):
            continue
    return sorted(results, key=lambda x: x['run_id'], reverse=True)


def undo_run(cfg, run_id: str = 'latest') -> dict:
    runs = get_runs(cfg)
    active = [r for r in runs if not r.get('undone')]
    if not active:
        raise ValueError('No recorded edits to undo')
    if run_id == 'latest':
        selected = active[0]
    else:
        selected = next((r for r in active if r['run_id'] == run_id), None)
        if selected is None:
            raise ValueError('Run not found or already undone')
        if selected['run_id'] != active[0]['run_id']:
            raise ValueError('Undo the newest outstanding run first to avoid overwriting later edits')

    # Preflight every edit in reverse order, including multiple edits to one file.
    virtual = {}
    restores = []
    for edit in reversed(selected['edits']):
        target = cfg.file(edit['file'])
        current = virtual.get(target)
        if current is None:
            current = target.read_bytes()
        if digest(current) != edit['after_sha256']:
            raise ValueError(f"File changed since agent edit: {edit['file']}; undo cancelled without changes")
        backup = Path(edit['backup']).resolve(strict=True)
        if not backup.is_relative_to(cfg.backup_root.resolve()):
            raise ValueError('Backup is outside the configured backup directory')
        original = backup.read_bytes()
        if digest(original) != edit['before_sha256']:
            raise ValueError(f"Backup checksum mismatch: {edit['file']}")
        virtual[target] = original
        restores.append((target, original))

    # Write one final original per file, preserving the current file mode.
    for target, original in virtual.items():
        atomic_write(target, original, target.stat().st_mode & 0o777)

    selected['undone'] = True
    selected['undone_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_write(journal_path(cfg, selected['run_id']), json.dumps(selected, indent=2).encode('utf-8'))
    return {'run_id': selected['run_id'], 'restored_files': [str(p.relative_to(cfg.root)) for p in virtual]}

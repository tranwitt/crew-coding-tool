"""Repository facts gathered by the host, so no model needs a tool call to get them."""
import ast
import difflib
import os
import re
from pathlib import Path

from config import EXCLUDED

OUTLINE_CHARS = 12000
LISTED_FILES = 400

_PATH = re.compile(r'(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.py(?![\w/.-])')
_FILE_LINE = re.compile(r'(?im)^[\s>*\-\d.)]*\**FILE\**\s*:\s*(.+)$')
_NEW_LINE = re.compile(r'(?im)^[\s>*\-\d.)]*\**NEW\**\s*:\s*(.+)$')
_STOPWORDS = frozenset((
    'the and for with from that this into add adds use using make change changes '
    'update fix file files code python function class method should when not all '
    'any are was has have support new').split())


def list_python_files(cfg, *, for_edit=True, limit=5000):
    """Project-relative .py files the run may edit (or, with for_edit=False, read)."""
    files = []
    for directory, dirs, names in os.walk(cfg.root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            if not name.endswith('.py'):
                continue
            relative = (Path(directory) / name).relative_to(cfg.root).as_posix()
            try:
                cfg.file(relative, for_edit=for_edit)
            except (ValueError, OSError):
                continue
            files.append(relative)
            if len(files) >= limit:
                return files
    return files


def _symbols(cfg, filename):
    """Return (line_count, top-level definition names) for a listing entry."""
    try:
        path = cfg.file(filename)
        if path.stat().st_size > 300_000:
            return None, []
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text)
    except (ValueError, OSError, UnicodeError, SyntaxError):
        return None, []
    names = [('class ' if isinstance(node, ast.ClassDef) else 'def ') + node.name
             for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return len(text.splitlines()), names


def describe_repository(cfg, files):
    """Compact file listing with top-level symbols, bounded for small context windows."""
    lines, used, symbols = [], 0, {}
    for filename in files[:LISTED_FILES]:
        entry = filename
        if used < OUTLINE_CHARS:
            count, names = _symbols(cfg, filename)
            symbols[filename] = names
            if count is not None:
                entry += f' ({count} lines)'
            if names:
                entry += ': ' + ', '.join(names[:12]) + (', ...' if len(names) > 12 else '')
        used += len(entry) + 1
        lines.append(entry)
    if len(files) > LISTED_FILES:
        lines.append(f'... and {len(files) - LISTED_FILES} more files not listed')
    return '\n'.join(lines), symbols


def files_named_in(text, cfg, limit, *, for_edit=True):
    """Existing Python paths cited in ``text``; nothing is ever invented.

    ``FILE: path`` lines are preferred when present, so a model that echoes the
    repository listing does not select every file it repeated.
    """
    def scan(part):
        matches = []
        for match in _PATH.finditer(part):
            name = match.group(0)
            while name.startswith('./'):
                name = name[2:]
            if name in matches:
                continue
            try:
                cfg.file(name, for_edit=for_edit)
            except (ValueError, OSError):
                continue
            matches.append(name)
            if len(matches) == limit:
                break
        return matches

    marked = '\n'.join(_FILE_LINE.findall(text))
    return (scan(marked) if marked else []) or scan(text)


def new_files_named_in(text, cfg, limit):
    """Paths on ``NEW: path`` lines that do not exist yet and may be created.

    Only explicitly marked lines count, and each path passes the same rules as
    an edit (inside the project, .py, not protected, within --allowed-paths).
    """
    names = []
    for line in _NEW_LINE.findall(text):
        match = _PATH.search(line)
        if not match or len(names) == limit:
            continue
        name = match.group(0)
        while name.startswith('./'):
            name = name[2:]
        try:
            path = cfg.file(name, must_exist=False, for_edit=True)
        except (ValueError, OSError):
            continue
        if name not in names and not path.exists():
            names.append(name)
    return names


def files_matching_request(request, files, symbols, limit):
    """Last resort when the planner names nothing usable: rank by word overlap."""
    words = {w for w in re.findall(r'[a-z_][a-z0-9_]{2,}', request.lower()) if w not in _STOPWORDS}
    scored = []
    for filename in files:
        path_words = set(re.findall(r'[a-z0-9]+', filename.lower()))
        name_words = set(re.findall(r'[a-z0-9]{4,}', ' '.join(symbols.get(filename, [])).lower()))
        # Prefix match in either direction so "greeting" finds def greet().
        score = 3 * len(words & path_words) + sum(
            1 for w in words for n in name_words - {'class'} if w.startswith(n) or n.startswith(w))
        if score:
            scored.append((-score, filename))
    chosen = [filename for _, filename in sorted(scored)[:limit]]
    if not chosen and len(files) <= 3:
        chosen = list(files)
    return chosen


def run_diffs(cfg, edits, budget=20000):
    """Unified diff per file for a run's journal entries: first backup vs. file now.

    A file the run created has no backup and is diffed against nothing.
    """
    originals = {}
    for edit in edits:
        originals.setdefault(edit['file'], edit['backup'])
    parts = []
    for filename, backup in originals.items():
        try:
            before = Path(backup).read_bytes().decode('utf-8') if backup else ''
            after = cfg.file(filename).read_bytes().decode('utf-8')
        except (ValueError, OSError, UnicodeError) as exc:
            parts.append(f'{filename}: diff unavailable ({exc})\n')
            continue
        parts.append(''.join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile=(filename + ' (before)') if backup else '/dev/null (new file)',
            tofile=filename + ' (after)')))
    text = '\n'.join(parts)
    return text if len(text) <= budget else text[:budget] + '\n... (diff truncated)\n'

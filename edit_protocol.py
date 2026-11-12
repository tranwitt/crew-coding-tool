"""Model-agnostic edit protocol.

Models differ wildly in how reliably they call tools or emit strict JSON, but
every chat model can return text. This module turns a plain-text answer into
exact (old_text, new_text) replacements that the journaled patch tool can apply.

Accepted answer formats, in order of preference:

1. SEARCH/REPLACE blocks (no escaping required, so the most reliable format)
2. A JSON object with a ``patches`` (or ``edits``) array, anywhere in the answer
3. The complete updated file in a single fenced code block
4. The literal sentinel ``NO_CHANGES_NEEDED``

Matching is tolerant of trailing whitespace, line endings and uniform
indentation shifts, but a match must always be unique. Nothing here writes
files: path checks, syntax checks, backups and the undo journal stay in
``patch_project_file``.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

NO_CHANGE_SENTINEL = 'NO_CHANGES_NEEDED'


class EditError(ValueError):
    """The answer could not be turned into a safe, unambiguous edit."""


@dataclass(frozen=True)
class Edit:
    old_text: str
    new_text: str
    kind: str  # 'search_replace', 'json' or 'whole_file'


def format_instructions(filename: str) -> str:
    """Prompt text describing the answer format; contains no tool requirements."""
    return (
        'Reply with your edits as SEARCH/REPLACE blocks in exactly this form, '
        'one block per change:\n\n'
        f'{filename}\n'
        '<<<<<<< SEARCH\n'
        'lines copied exactly from the current file\n'
        '=======\n'
        'the lines that replace them\n'
        '>>>>>>> REPLACE\n\n'
        'Rules:\n'
        '- SEARCH must be whole lines copied from the current file and long enough '
        'to occur only once; add unchanged neighbouring lines if needed.\n'
        '- To insert code, put an existing neighbouring line in SEARCH and repeat '
        'it in REPLACE together with the new lines.\n'
        '- Keep the indentation of the file. Never abbreviate code with "..." or '
        'comments such as "rest unchanged".\n'
        '- Do not describe a change instead of making it. Text outside the blocks '
        'is ignored.\n'
        f'- If this file needs no change, reply with exactly {NO_CHANGE_SENTINEL}.\n'
        'Also accepted: one JSON object {"patches":[{"filename":"' + filename + '",'
        '"old_text":"...","new_text":"..."}]}, or, for a short file, the complete '
        'updated file in a single ```python block.'
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_SEARCH = re.compile(r'^\s*<{5,9}\s*SEARCH\s*$')
_DIVIDER = re.compile(r'^\s*={5,9}\s*$')
_REPLACE = re.compile(r'^\s*>{5,9}\s*REPLACE\s*$')
_PATH_LINE = re.compile(r'[\w./\\-]+\.\w+')
_ELISION = re.compile(
    r'^\s*(?:#|//)?\s*\.\.\.\s*\(?\s*(?:the\s+)?'
    r'(?:rest|existing|remaining|unchanged|same|previous|other)\b'
    r'|^\s*#\s*(?:rest of|existing|remaining)\b.*\b(?:unchanged|omitted|here|as before)\b',
    re.I)


def _same_file(candidate: str, filename: str) -> bool:
    candidate = candidate.replace('\\', '/')
    while candidate.startswith('./'):
        candidate = candidate[2:]
    return candidate == filename or candidate.endswith('/' + filename)


def _named_file(lines, index):
    """Return the path written on the line above a SEARCH marker, if any."""
    for line in reversed(lines[max(0, index - 3):index]):
        text = line.strip()
        if not text or text.startswith('```'):
            continue
        text = re.sub(r'^(?:#+\s*|file\s*:\s*|path\s*:\s*)', '', text, flags=re.I)
        text = text.strip('`*: \t')
        return text if _PATH_LINE.fullmatch(text) else None
    return None


def _check_elision(new_text: str, source: str, where: str):
    for line in new_text.splitlines():
        if _ELISION.search(line) and line.strip() not in source:
            raise EditError(
                f'{where} abbreviates code with a placeholder ({line.strip()[:60]!r}). '
                'Write out every line that must be in the file.')


def _parse_search_replace(raw: str, filename: str, source: str):
    lines = raw.splitlines()
    edits = []
    i = 0
    while i < len(lines):
        if not _SEARCH.match(lines[i]):
            i += 1
            continue
        number = len(edits) + 1
        named = _named_file(lines, i)
        if named and not _same_file(named, filename):
            raise EditError(f'Block {number} targets {named!r}; only {filename!r} may be edited in this task')
        divider = next((j for j in range(i + 1, len(lines)) if _DIVIDER.match(lines[j])), None)
        if divider is None:
            raise EditError(f'Block {number} has no ======= divider line')
        end = next((j for j in range(divider + 1, len(lines)) if _REPLACE.match(lines[j])), None)
        if end is None:
            raise EditError(f'Block {number} has no >>>>>>> REPLACE line')
        old = '\n'.join(lines[i + 1:divider])
        new = '\n'.join(lines[divider + 1:end])
        if not old.strip():
            raise EditError(
                f'Block {number} has an empty SEARCH section; copy an existing '
                'neighbouring line into SEARCH and repeat it in REPLACE')
        if old == new:
            raise EditError(f'Block {number} does not change anything')
        _check_elision(new, source, f'Block {number}')
        edits.append(Edit(old, new, 'search_replace'))
        i = end + 1
    return edits


def _json_candidates(raw: str):
    decoder = json.JSONDecoder()
    position = raw.find('{')
    tried = 0
    while position != -1 and tried < 200:
        tried += 1
        try:
            value, length = decoder.raw_decode(raw[position:])
        except ValueError:
            position = raw.find('{', position + 1)
            continue
        yield value
        position = raw.find('{', position + length)


def _parse_json(raw: str, filename: str):
    if '"patches"' not in raw and '"edits"' not in raw:
        return []
    for value in _json_candidates(raw):
        if not isinstance(value, dict):
            continue
        items = value.get('patches', value.get('edits'))
        if not isinstance(items, list):
            continue
        if not items:
            raise EditError('The patches array is empty')
        edits = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise EditError(f'patches[{index}] must be an object')
            old = item.get('old_text', item.get('search'))
            new = item.get('new_text', item.get('replace'))
            named = item.get('filename', item.get('file'))
            if named is not None and not (isinstance(named, str) and _same_file(named, filename)):
                raise EditError(f'patches[{index}] targets {named!r}; only {filename!r} may be edited in this task')
            if not isinstance(old, str) or not old:
                raise EditError(f'patches[{index}].old_text must be a nonempty string')
            if not isinstance(new, str) or new == old:
                raise EditError(f'patches[{index}].new_text must be a different string')
            edits.append(Edit(old, new, 'json'))
        return edits
    return []


def _fenced_blocks(raw: str):
    blocks, language, body = [], None, []
    for line in raw.splitlines():
        if line.strip().startswith('```'):
            if language is None:
                language, body = line.strip()[3:].strip().lower(), []
            else:
                blocks.append((language, '\n'.join(body)))
                language = None
        elif language is not None:
            body.append(line)
    return blocks


def _parse_whole_file(raw: str, filename: str, source: str):
    """Return an Edit, the string 'unchanged', or None when the format is absent."""
    blocks = [(lang, body) for lang, body in _fenced_blocks(raw)
              if lang in ('', 'python', 'py', 'python3') and body.strip()]
    if len(blocks) != 1:
        return None
    language, body = blocks[0]
    try:
        ast.parse(body)
    except SyntaxError as exc:
        if not language:
            return None
        raise EditError(f'The code block is not valid Python ({exc.msg}, line {exc.lineno}). '
                        'Use SEARCH/REPLACE blocks instead.') from exc
    original_lines = source.count('\n') + 1
    if original_lines > 10 and body.count('\n') + 1 < 0.6 * original_lines:
        raise EditError(
            f'The answer is one code block of {body.count(chr(10)) + 1} lines but {filename} '
            f'has {original_lines}; it looks like a fragment. Use SEARCH/REPLACE blocks so '
            'the host knows which lines to replace.')
    _check_elision(body, source, 'The code block')
    if '\r\n' in source:
        body = body.replace('\n', '\r\n')
    if source.endswith('\n'):
        body = body.rstrip('\r\n') + ('\r\n' if '\r\n' in source else '\n')
    if body == source:
        return 'unchanged'
    return Edit(source, body, 'whole_file')


def parse_edits(raw, filename: str, source: str):
    """Return ``(edits, no_change)`` parsed from a model answer.

    Raises EditError with a message suitable for feeding back to the model.
    """
    raw = str(raw or '')
    edits = _parse_search_replace(raw, filename, source) or _parse_json(raw, filename)
    if edits:
        return edits, False
    whole = _parse_whole_file(raw, filename, source)
    if isinstance(whole, Edit):
        return [whole], False
    if whole == 'unchanged' or NO_CHANGE_SENTINEL in raw:
        return [], True
    raise EditError(
        'The answer contains no edits in an accepted format. A description or a bare '
        'code snippet does not change the file: reply with SEARCH/REPLACE blocks '
        f'(or exactly {NO_CHANGE_SENTINEL}).')


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _trim_blank_edges(old_lines, new_lines):
    """Drop blank lines framing SEARCH, and the matching ones framing REPLACE."""
    for index in (0, -1):
        while old_lines and not old_lines[index].strip():
            old_lines.pop(index)
            if new_lines and not new_lines[index].strip():
                new_lines.pop(index)


def _find_block(source_lines, wanted, normalise):
    target = [normalise(line) for line in wanted]
    normalised = [normalise(line) for line in source_lines]
    size = len(target)
    return [i for i in range(len(normalised) - size + 1) if normalised[i:i + size] == target]


def _contains(outer, inner):
    size = len(inner)
    return any(outer[i:i + size] == inner for i in range(len(outer) - size + 1))


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _reindent(new_lines, wanted_indent: str, given_indent: str):
    if wanted_indent == given_indent:
        return new_lines
    if wanted_indent.startswith(given_indent):
        extra = wanted_indent[len(given_indent):]
        return [extra + line if line.strip() else line for line in new_lines]
    if given_indent.startswith(wanted_indent):
        extra = given_indent[len(wanted_indent):]
        return [line[len(extra):] if line.startswith(extra) else line for line in new_lines]
    return new_lines


_NORMALISERS = (
    ('exact', lambda line: line.rstrip('\r\n')),
    ('trailing-whitespace', lambda line: line.rstrip()),
    ('indentation', lambda line: line.strip()),
)


def _make_unique(source: str, start: int, end: int, new: str):
    """Grow a line-aligned span until its text is unique for the patch tool."""
    for _ in range(200):
        if source.count(source[start:end]) == 1:
            return source[start:end], new
        grew = False
        if start > 0:
            previous = source.rfind('\n', 0, start - 1) + 1
            new, start, grew = source[previous:start] + new, previous, True
        if source.count(source[start:end]) != 1 and end < len(source):
            following = source.find('\n', end)
            following = len(source) if following == -1 else following + 1
            new, end, grew = new + source[end:following], following, True
        if not grew:
            break
    raise EditError('Could not isolate a unique location for this edit; include more surrounding lines')


def resolve_edit(source: str, edit: Edit):
    """Map an Edit onto the current file text.

    Returns ``(old_text, new_text)`` where old_text occurs exactly once in
    ``source``, or ``None`` when the change is already present. Raises EditError
    when the location is missing or ambiguous.
    """
    if edit.kind == 'whole_file':
        return (source, edit.new_text) if edit.new_text != source else None

    if edit.kind == 'json':
        # JSON patches may replace a fragment inside a line; honour exact text first.
        count = source.count(edit.old_text)
        if count == 1:
            return edit.old_text, edit.new_text
        if count > 1:
            raise EditError(f'old_text matches {count} places; include more surrounding text so it is unique')

    old_lines, new_lines = edit.old_text.splitlines(), edit.new_text.splitlines()
    _trim_blank_edges(old_lines, new_lines)
    if not old_lines:
        raise EditError('SEARCH text is empty')
    source_lines = source.splitlines(keepends=True)
    rstrip = _NORMALISERS[1][1]

    for name, normalise in _NORMALISERS:
        hits = _find_block(source_lines, old_lines, normalise)
        if len(hits) > 1:
            raise EditError(f'SEARCH text matches {len(hits)} places; include more surrounding lines so it is unique')
        if not hits:
            continue
        first = hits[0]
        matched = source_lines[first:first + len(old_lines)]
        if name == 'indentation':
            anchor = next(i for i, line in enumerate(old_lines) if line.strip())
            new_lines = _reindent(new_lines, _indent(matched[anchor]), _indent(old_lines[anchor]))
        # An insertion that repeats its anchor is already applied if the result is in the file.
        if (len(new_lines) > len(old_lines)
                and _contains([rstrip(l) for l in new_lines], [rstrip(l) for l in old_lines])
                and _find_block(source_lines, new_lines, rstrip)):
            return None
        old_text = ''.join(matched)
        eol = '\r\n' if '\r\n' in old_text else '\n'
        new_text = eol.join(new_lines)
        if new_lines and old_text.endswith('\n'):
            new_text += eol
        if new_text == old_text:
            return None
        start = sum(len(line) for line in source_lines[:first])
        return _make_unique(source, start, start + len(old_text), new_text)

    if any(line.strip() for line in new_lines) and _find_block(source_lines, new_lines, rstrip):
        return None  # A previous attempt already made this change.
    first_line = next(line.strip() for line in old_lines if line.strip())
    raise EditError(
        f'SEARCH text was not found in the current file (it starts with {first_line[:80]!r}). '
        'Copy the lines exactly from the file content shown in the task.')

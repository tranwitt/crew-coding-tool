"""CrewAI-first focused implementation with journal-backed completion checks.

Every model invocation happens inside Crew.kickoff(), but no stage depends on
the model being able to call tools:

* The host lists the repository and embeds the current file in each task, so a
  model never has to call a tool to see what it is editing.
* A model may patch through its tools, or simply answer in text. Text answers
  are parsed by edit_protocol and applied through the same journaled patch tool.
* An attempt that fails with tools is repeated without tools.

An edit only counts when it appears in the undo journal.
"""
import ast
import json
import os
import re
from pathlib import Path

from config import EXCLUDED
from edit_protocol import EditError, format_instructions, parse_edits, resolve_edit

PLAN_CONTEXT_CHARS = 4000
OUTLINE_CHARS = 12000
LISTED_FILES = 400


def _raw(result):
    return str(getattr(result, 'raw', None) or result)


def _usage(result):
    u = getattr(result, 'token_usage', None)
    if u is None:
        return {}
    def get(*keys):
        for k in keys:
            v = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
            if isinstance(v, (int, float)):
                return int(v)
        return 0
    return {'prompt': get('prompt_tokens', 'input_tokens'),
            'completion': get('completion_tokens', 'output_tokens'),
            'total': get('total_tokens')}


def _parse_patches(raw, filename):
    """Strict schema validation. Never silently discard malformed edits."""
    cleaned = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        return [], f'Output is not a single valid JSON object: {exc}'
    if not isinstance(data, dict) or set(data) != {'patches'}:
        return [], 'Expected exactly one object with key "patches"'
    edits = data['patches']
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        return [], 'patches must be a nonempty array of at most 8 edits'
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict) or set(edit) != {'filename', 'old_text', 'new_text'}:
            return [], f'patches[{i}] requires exactly filename, old_text, new_text'
        if edit['filename'] != filename:
            return [], f'patches[{i}] targets {edit["filename"]!r}, expected {filename!r}'
        if not isinstance(edit['old_text'], str) or not edit['old_text']:
            return [], f'patches[{i}].old_text must be a nonempty string'
        if not isinstance(edit['new_text'], str) or edit['old_text'] == edit['new_text']:
            return [], f'patches[{i}].new_text must be a different string'
    return edits, None


def _json_patches(raw, filename):
    """Backward-compatible parser for callers expecting a list."""
    return _parse_patches(raw, filename)[0]


def _apply_patches(patches, filename, patch_tool, edits_in_run):
    """Apply validated patches; require a successful journal entry for every patch."""
    applied = 0
    for i, patch in enumerate(patches):
        before = len(edits_in_run())
        try:
            response = patch_tool.run(**patch)
        except Exception as exc:
            return applied, f'patches[{i}] raised {type(exc).__name__}: {exc}'
        if len(edits_in_run()) != before + 1:
            return applied, f'patches[{i}] was not journaled: {str(response)[:500]}'
        applied += 1
    return applied, None

def _apply_edits(edits, filename, read_source, patch_tool, edits_in_run):
    """Apply parsed edits one by one against the file as it is now.

    Returns ``(applied, already_present, error)``. Each edit is located in the
    current text first, so the patch tool always receives an exact unique match.
    """
    applied = present = 0
    total = len(edits)
    for i, edit in enumerate(edits, 1):
        try:
            resolved = resolve_edit(read_source(), edit)
        except EditError as exc:
            return applied, present, f'edit {i} of {total}: {exc}'
        if resolved is None:
            present += 1
            continue
        before = len(edits_in_run())
        try:
            response = patch_tool.run(filename=filename, old_text=resolved[0], new_text=resolved[1])
        except Exception as exc:
            return applied, present, f'edit {i} of {total} raised {type(exc).__name__}: {exc}'
        if len(edits_in_run()) != before + 1:
            return applied, present, f'edit {i} of {total} was rejected: {str(response)[:500]}'
        applied += 1
    return applied, present, None


_PATH = re.compile(r'(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.py(?![\w/.-])')
_FILE_LINE = re.compile(r'(?im)^[\s>*\-\d.)]*\**FILE\**\s*:\s*(.+)$')


def _selected_files(plan, cfg, limit):
    """Only accept explicit existing Python paths cited in the planner output.

    ``FILE: path`` lines are preferred when present, so a model that echoes the
    repository listing does not select every file it repeated.
    """
    def scan(text):
        matches = []
        for match in _PATH.finditer(text):
            name = match.group(0)
            while name.startswith('./'):
                name = name[2:]
            if name in matches:
                continue
            try:
                cfg.file(name, for_edit=True)
            except (ValueError, OSError):
                continue
            matches.append(name)
            if len(matches) == limit:
                break
        return matches

    marked = '\n'.join(_FILE_LINE.findall(plan))
    return (scan(marked) if marked else []) or scan(plan)


def list_editable_files(cfg, limit=5000):
    """Project-relative editable .py files, found without any model tool call."""
    files = []
    for directory, dirs, names in os.walk(cfg.root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            if not name.endswith('.py'):
                continue
            relative = (Path(directory) / name).relative_to(cfg.root).as_posix()
            try:
                cfg.file(relative, for_edit=True)
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


_STOPWORDS = frozenset((
    'the and for with from that this into add adds use using make change changes '
    'update fix file files code python function class method should when not all '
    'any are was has have support new').split())


def _heuristic_files(request, files, symbols, limit):
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


def run_focused_crew(*, cfg, request, agents, task_defs, render, tool_lookup,
                     max_files, allow_tests, edits_in_run, Crew, Task, Process,
                     text_agents=None, tool_modes=None, max_inline_chars=60000,
                     extra_attempts=0):
    """Plan, then edit one file per CrewAI task, whatever the model can do.

    ``agents`` have their YAML tools; ``text_agents`` are the same agents with no
    tools. ``tool_modes`` maps 'planner'/'developer' to 'tools' or 'text'.
    """
    text_agents = text_agents or {}
    tool_modes = dict(tool_modes or {})
    if agents.get('planner') is None or agents.get('developer') is None:
        raise ValueError('Focused workflow requires planner and developer agents in agents.yaml')

    def mode_for(name):
        return 'text' if tool_modes.get(name) == 'text' and name in text_agents else 'tools'

    usage = {}
    last_result = [None]

    def kickoff(name, mode, description, expected_output):
        """One CrewAI task. Returns (raw_output, error); provider errors do not abort the run."""
        agent = text_agents[name] if mode == 'text' and name in text_agents else agents[name]
        task = Task(description=description, expected_output=expected_output, agent=agent)
        try:
            result = Crew(agents=[agent], tasks=[task],
                          process=Process.sequential, verbose=True).kickoff()
        except Exception as exc:
            return '', f'{type(exc).__name__}: {str(exc)[:400]}'
        last_result[0] = result
        for k, v in _usage(result).items():
            usage[k] = usage.get(k, 0) + v
        return _raw(result), None

    files = list_editable_files(cfg)
    listing, symbols = describe_repository(cfg, files)

    def plan_once(mode):
        tools_line = ('You have no tools in this task and need none: do not attempt tool calls.'
                      if mode == 'text' else
                      'You may use your read-only tools to inspect files before answering.')
        description = (
            f'Repository: {cfg.root}\nRequest: {request}\n{tools_line}\n'
            'The host application listed the editable Python files (with their top-level '
            f'definitions) for you:\n<files>\n{listing}\n</files>\n'
            'Decide which of these existing files must be edited. Answer with one line per '
            'file, most important first, in exactly this form:\n'
            'FILE: path/to/file.py - one sentence describing the change\n'
            'Then add essential interface constraints as short bullet points. Use only '
            'paths from the list. Limit to 600 words. Do not write code and do not edit.')
        print(f'CREWAI STAGE: planning ({mode}) via Crew.kickoff()', flush=True)
        return kickoff('planner', mode, description,
                       'FILE: lines naming existing Python files, then brief constraints.')

    plan, selected, source_of_selection = '', [], 'planner'
    planner_modes = (['tools', 'text'] if mode_for('planner') == 'tools' and 'planner' in text_agents
                     else [mode_for('planner')])
    for mode in planner_modes:
        plan, error = plan_once(mode)
        if error:
            print(f'CREWAI PLANNING FAILURE ({mode}): {error}', flush=True)
        selected = _selected_files(plan, cfg, max_files)
        if selected:
            break
    if not selected:
        selected = _selected_files(request, cfg, max_files)
        source_of_selection = 'paths named in request'
    if not selected:
        selected = _heuristic_files(request, files, symbols, min(max_files, 3))
        source_of_selection = 'keyword match (planner named no valid file)'
    print(f'CREWAI FILE TARGETS ({source_of_selection}): ' + json.dumps(selected), flush=True)
    if not selected:
        return last_result[0], {'mode': 'focused-crewai', 'selected_files': [],
                                'incomplete_files': [], 'tool_modes': tool_modes,
                                'error': 'No editable Python file could be identified for the request',
                                'usage': usage}

    patch_tool = tool_lookup['patch_project_file']
    plan_context = plan.strip()[:PLAN_CONTEXT_CHARS]
    developer_mode = mode_for('developer')
    completed, incomplete, unchanged, reports = [], [], [], []

    for filename in selected:
        def read_source():
            # Bytes, not text mode: the patch tool matches against undecoded newlines.
            return cfg.file(filename, for_edit=True).read_bytes().decode('utf-8')

        file_start = len(edits_in_run())
        errors, attempts = [], []
        no_change = False
        feedback = ''
        modes = ([developer_mode] if developer_mode == 'text' else ['tools', 'text']) + ['text'] * (1 + extra_attempts)
        if 'developer' not in text_agents:
            modes = ['tools'] * len(modes)
        for number, mode in enumerate(modes, 1):
            try:
                source = read_source()
            except (ValueError, OSError, UnicodeError) as exc:
                errors.append(f'Cannot read {filename}: {exc}')
                attempts.append({'attempt': number, 'mode': mode, 'result': 'failed', 'error': errors[-1]})
                break
            inline = len(source) <= max_inline_chars
            if not inline and mode == 'text':
                errors.append(f'{filename} has {len(source)} characters, above --max-inline-chars '
                              f'({max_inline_chars}); it cannot be edited without tools')
                attempts.append({'attempt': number, 'mode': mode, 'result': 'failed', 'error': errors[-1]})
                print(f'CREWAI VALIDATION FAILURE {filename}: {errors[-1]}', flush=True)
                break
            if mode == 'text':
                how = ('You have NO tools in this task. Do not attempt tool calls and ignore any '
                       'instruction to call one: the host application applies the edits you '
                       'return and verifies them.')
            else:
                how = ('You may call patch_project_file(filename, old_text, new_text) directly. '
                       'Unless that tool confirms "status": "updated", your final answer must '
                       'contain the edits in the format below; the host application applies them.')
            content = (f'Current content of {filename}:\n<current_file>\n{source}\n</current_file>\n'
                       if inline else
                       f'{filename} is too large to show here; read it with read_project_file.\n')
            description = (
                f'Repository: {cfg.root}\nUser request: {request}\n'
                + (f'Plan from the architect (context only; other files are handled separately):\n'
                   f'<plan>\n{plan_context}\n</plan>\n' if plan_context else '')
                + f'Edit ONLY {filename}. {how}\n{content}'
                + format_instructions(filename)
                + '\nNever claim a file was modified. Do not run tests.' + feedback)
            before = len(edits_in_run())
            print(f'CREWAI STAGE: edit {filename} attempt {number}/{len(modes)} ({mode}) via Crew.kickoff()', flush=True)
            raw, error = kickoff('developer', mode, description,
                                 'SEARCH/REPLACE blocks for the file (or a confirmed patch tool result). No narrative.')
            by_tool = len(edits_in_run()) - before
            applied = present = 0
            if by_tool:
                error = None  # The journal, not the narrative, is the evidence.
            elif error is None:
                try:
                    edits, no_change = parse_edits(raw, filename, source)
                except EditError as exc:
                    edits, error = [], str(exc)
                if edits:
                    applied, present, error = _apply_edits(edits, filename, read_source,
                                                           patch_tool, edits_in_run)
                    print(f'CREWAI TEXT EDITS {filename}: {applied} journaled, '
                          f'{present} already present', flush=True)
            if mode == 'tools' and not by_tool and developer_mode == 'tools':
                developer_mode = 'text'
                print('CREWAI TOOL MODE: the developer model did not patch through its tools; '
                      'remaining edits use the tool-free text protocol.', flush=True)
            attempts.append({'attempt': number, 'mode': mode,
                             'result': 'failed' if error else ('no_change' if no_change else 'journal_verified'),
                             'journaled_by_tool': by_tool, 'journaled_from_text': applied,
                             'already_present': present, 'error': error})
            if error is None:
                break
            errors.append(error)
            print(f'CREWAI VALIDATION FAILURE {filename}: {error}', flush=True)
            feedback = ('\nPREVIOUS ATTEMPT FAILED VALIDATION: ' + error[:700] +
                        '\nThe file content above is CURRENT'
                        + (f' and already includes {applied} edit(s) from your previous answer'
                           if applied else '') +
                        '. Return only the edits that are still missing; do not repeat applied ones.')
        count = len(edits_in_run()) - file_start
        clean = bool(attempts) and attempts[-1]['error'] is None
        complete = bool(count) and clean
        if complete:
            completed.append(filename)
        elif clean:
            unchanged.append(filename)
        else:
            incomplete.append(filename)
        reports.append({'file': filename, 'verified': bool(count), 'complete': complete,
                        'unchanged': clean and not count,
                        'journaled_operations': count, 'attempts': attempts, 'errors': errors})
        print(f'CREWAI VERIFIED {filename}: {count} journaled operation(s); '
              f'complete={complete}', flush=True)

    test_status = 'not run (--allow-tests disabled)'
    if allow_tests:
        test_targets = [f for f in selected if f.startswith('tests/') and Path(f).name.startswith('test_')]
        target = test_targets[0] if test_targets else 'tests'
        try:
            test_status = str(tool_lookup['run_pytest'].run(test_path=target))[:3000]
        except Exception as exc:
            test_status = f'ERROR: {exc}'
        print('TEST VERIFICATION: ' + test_status, flush=True)

    tool_modes['developer_final'] = developer_mode
    status = {'mode': 'focused-crewai', 'tool_modes': tool_modes,
              'selected_files': selected, 'selection_source': source_of_selection,
              'completed_files': completed, 'incomplete_files': incomplete,
              'unchanged_files': unchanged,
              'reports': reports, 'tests': test_status, 'usage': usage}
    return last_result[0], status

"""CrewAI-first focused implementation with journal-backed completion checks.

Every model invocation happens inside Crew.kickoff(). Python only validates and
applies edits emitted by a CrewAI task if the agent did not call its patch tool.
"""
import json
import re
from pathlib import Path


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

def _selected_files(plan, cfg, limit):
    """Only accept explicit existing Python paths cited in the planner output."""
    matches = []
    seen = set()
    for match in re.finditer(r'(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.py(?![\w/.-])', plan):
        name = match.group(0)
        if name in seen:
            continue
        try:
            cfg.file(name, for_edit=True)
        except (ValueError, OSError):
            continue
        seen.add(name)
        matches.append(name)
        if len(matches) == limit:
            break
    return matches


def run_focused_crew(*, cfg, request, agents, task_defs, render, tool_lookup,
                     max_files, allow_tests, edits_in_run, Crew, Task, Process):
    planner = agents.get('planner')
    developer = agents.get('developer')
    if planner is None or developer is None:
        raise ValueError('Focused workflow requires planner and developer agents in agents.yaml')

    # One short planning CrewAI task. No large file contents carried into coding context.
    planning = Task(
        description=(f'Repository: {cfg.root}\nRequest: {request}\n'
                     'Use read-only repository tools to inspect only relevant files. '
                     'Return a BRIEF ordered list of existing project-relative .py files '
                     'that require editing, with one sentence per file and essential '
                     'interface constraints. Limit to 1200 words. Do not edit.'),
        expected_output='Concise ordered Python file list and minimal implementation constraints.',
        agent=planner,
    )
    print('CREWAI STAGE: planning via Crew.kickoff()', flush=True)
    plan_result = Crew(agents=[planner], tasks=[planning],
                       process=Process.sequential, verbose=True).kickoff()
    plan = _raw(plan_result)
    selected = _selected_files(plan, cfg, max_files)
    print('CREWAI FILE TARGETS: ' + json.dumps(selected), flush=True)
    if not selected:
        return plan_result, {'mode': 'focused-crewai', 'selected_files': [],
                             'incomplete_files': [], 'error': 'Planner did not identify editable Python paths',
                             'usage': _usage(plan_result)}

    completed = []
    incomplete = []
    reports = []
    usage = _usage(plan_result)
    for filename in selected:
        file_start = len(edits_in_run())
        errors = []
        attempts = []
        result = None
        for attempt in range(2):
            retry_note = '' if attempt == 0 else (
                '\nPREVIOUS ATTEMPT FAILED VALIDATION: ' + errors[-1][:700] +
                '\nInspect the CURRENT file again. Return only the missing edits. '
                'Do not repeat already applied changes. Do not claim success without a journaled patch.')
            task = Task(
                description=(f'Repository: {cfg.root}\nUser request: {request}\n'
                             f'Edit ONLY {filename}. Read the current file first. '
                             'Produce concrete, minimal edits to fulfill the request. '
                             'You may call patch_project_file directly. '
                             'If you do not successfully call the patch tool, your FINAL OUTPUT '
                             'MUST be exactly one JSON object, no prose, no Markdown, with schema '
                             '{"patches":[{"filename":"' + filename + '",'
                             '"old_text":"exact unique current source snippet",'
                             '"new_text":"replacement source snippet"}]}. '
                             'old_text must occur exactly once in the current file. '
                             'Never claim a file was modified unless the patch tool confirmed it. '
                             'Do not run tests.' + retry_note),
                expected_output=('A confirmed patch_project_file tool result, or a single strict '
                                 'JSON object with a nonempty patches array. No narrative.'),
                agent=developer,
            )
            print(f'CREWAI STAGE: edit {filename} attempt {attempt + 1}/2 via Crew.kickoff()', flush=True)
            result = Crew(agents=[developer], tasks=[task],
                          process=Process.sequential, verbose=True).kickoff()
            for k, v in _usage(result).items():
                usage[k] = usage.get(k, 0) + v
            raw = _raw(result)
            journal_delta = len(edits_in_run()) - file_start
            if journal_delta > 0:
                attempts.append({'attempt': attempt + 1, 'result': 'journal_verified',
                                 'journaled_operations': journal_delta})
                break
            patches, error = _parse_patches(raw, filename)
            if error is None:
                applied, error = _apply_patches(patches, filename,
                                               tool_lookup['patch_project_file'], edits_in_run)
                if applied:
                    print(f'CREWAI STRUCTURED PATCH {filename}: {applied} journaled edit(s)', flush=True)
            if error:
                errors.append(error)
                print(f'CREWAI VALIDATION FAILURE {filename}: {error}', flush=True)
            journal_delta = len(edits_in_run()) - file_start
            if journal_delta > 0:
                attempts.append({'attempt': attempt + 1, 'result': 'journal_verified',
                                 'journaled_operations': journal_delta,
                                 'remaining_error': error})
                # Do not retry after a partial successful application: a new attempt
                # could duplicate work. Report the partial failure explicitly.
                break
            attempts.append({'attempt': attempt + 1, 'result': 'failed', 'error': error})
        count = len(edits_in_run()) - file_start
        complete = bool(count) and not (attempts[-1].get('remaining_error') if attempts else True)
        if complete:
            completed.append(filename)
        else:
            incomplete.append(filename)
        reports.append({'file': filename, 'verified': bool(count),
                        'complete': complete,
                        'journaled_operations': count, 'attempts': attempts,
                        'errors': errors})
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

    status = {'mode': 'focused-crewai', 'selected_files': selected,
              'completed_files': completed, 'incomplete_files': incomplete,
              'reports': reports, 'tests': test_status, 'usage': usage}
    return result, status

"""The one execution pipeline: select -> edit -> verify -> explain, or select -> analyse.

Every model invocation happens inside Crew.kickoff(), but no stage depends on
the model being able to call tools:

* The host lists the repository and embeds file contents in each task.
* A model may patch through its tools, or simply answer in text. Text answers
  are parsed by edit_protocol and applied through the journaled patch tool.
* A stage that fails with tools is repeated without tools.

Which agent runs a stage, and what it is told, comes from tasks.yaml (see
task_config). An edit only counts when it appears in the undo journal.
"""
import json
from pathlib import Path

from edit_protocol import EditError, format_instructions, parse_edits, resolve_edit
from task_config import require_stages
from repo_context import (describe_repository, files_matching_request, files_named_in,
                          list_python_files, run_diffs)

PLAN_CONTEXT_CHARS = 4000


def _raw(result):
    return str(getattr(result, 'raw', None) or result)


def _usage(result):
    u = getattr(result, 'token_usage', None)
    if u is None:
        return {}
    def get(*keys):
        for k in keys:
            v = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return 0
    return {'prompt': get('prompt_tokens', 'input_tokens'),
            'completion': get('completion_tokens', 'output_tokens'),
            'total': get('total_tokens')}


class CrewRunner:
    """Runs single-agent CrewAI tasks and tracks tool modes and token usage.

    ``agents`` carry their YAML tools; ``text_agents`` are the same agents with
    none. ``tool_modes`` maps an agent name to 'tools' or 'text'.
    """
    def __init__(self, *, agents, text_agents, tool_modes, Crew, Task, Process):
        self.agents, self.text_agents = agents, text_agents or {}
        self.tool_modes = dict(tool_modes or {})
        self.Crew, self.Task, self.Process = Crew, Task, Process
        self.usage = {}

    def mode(self, name):
        return 'text' if self.tool_modes.get(name) == 'text' and name in self.text_agents else 'tools'

    def use_text(self, name):
        """Stop offering tools to an agent for the rest of the run."""
        if name in self.text_agents:
            self.tool_modes[name] = 'text'

    def modes_to_try(self, name):
        """The agent's mode, followed by the tool-free fallback when it has one."""
        return ['tools', 'text'] if self.mode(name) == 'tools' and name in self.text_agents else [self.mode(name)]

    def ask(self, name, mode, description, expected_output):
        """One CrewAI task. Returns (raw_output, error); provider errors do not abort the run."""
        agent = self.text_agents[name] if mode == 'text' and name in self.text_agents else self.agents[name]
        task = self.Task(description=description, expected_output=expected_output, agent=agent)
        try:
            result = self.Crew(agents=[agent], tasks=[task],
                               process=self.Process.sequential, verbose=True).kickoff()
        except Exception as exc:
            return '', f'{type(exc).__name__}: {str(exc)[:400]}'
        for key, value in _usage(result).items():
            self.usage[key] = self.usage.get(key, 0) + value
        return _raw(result), None


# --------------------------------------------------------------------------
# Stage: select files
# --------------------------------------------------------------------------

def select_files(runner, tasks, cfg, request, limit, *, for_edit):
    """Ask the select stage which files matter. Returns (files, plan, source, listing)."""
    stage = tasks['select']
    files = list_python_files(cfg, for_edit=for_edit)
    listing, symbols = describe_repository(cfg, files)
    plan, selected, source = '', [], stage.agent
    for mode in runner.modes_to_try(stage.agent):
        print(f'CREWAI STAGE: select files ({mode}) via Crew.kickoff()', flush=True)
        plan, error = runner.ask(stage.agent, mode, stage.render(
            mode, repo=cfg.root, request=request, listing=listing,
            file_scope='editable Python files' if for_edit else 'Python files',
            purpose='must be edited' if for_edit else 'must be read to answer the request'),
            stage.expected_output)
        if error:
            print(f'CREWAI SELECTION FAILURE ({mode}): {error}', flush=True)
        selected = files_named_in(plan, cfg, limit, for_edit=for_edit)
        if selected:
            break
    if not selected:
        selected, source = files_named_in(request, cfg, limit, for_edit=for_edit), 'paths named in request'
    if not selected:
        selected = files_matching_request(request, files, symbols, min(limit, 3))
        source = f'keyword match ({stage.agent} named no valid file)'
    print(f'CREWAI FILE TARGETS ({source}): ' + json.dumps(selected), flush=True)
    return selected, plan, source, listing


# --------------------------------------------------------------------------
# Stage: edit one file
# --------------------------------------------------------------------------

def apply_edits(edits, filename, read_source, patch_tool, edits_in_run):
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


def edit_file(runner, tasks, cfg, request, filename, plan_context, *, patch_tool, edits_in_run,
              max_inline_chars, extra_attempts):
    """Run edit-stage attempts on one file until an attempt validates. Returns its report."""
    stage = tasks['edit']
    def read_source():
        # Bytes, not text mode: the patch tool matches against undecoded newlines.
        return cfg.file(filename, for_edit=True).read_bytes().decode('utf-8')

    file_start = len(edits_in_run())
    errors, attempts = [], []
    feedback = ''
    modes = runner.modes_to_try(stage.agent)
    modes += [modes[-1]] * (1 + extra_attempts)
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
        content = (f'Current content of {filename}:\n<current_file>\n{source}\n</current_file>'
                   if inline else
                   f'{filename} is too large to show here; read it with read_project_file.')
        description = stage.render(
            mode, repo=cfg.root, request=request, filename=filename, file_content=content,
            plan=plan_context or '(no plan available)',
            edit_format=format_instructions(filename), feedback=feedback)
        before = len(edits_in_run())
        print(f'CREWAI STAGE: edit {filename} attempt {number}/{len(modes)} ({mode}) via Crew.kickoff()', flush=True)
        raw, error = runner.ask(stage.agent, mode, description, stage.expected_output)
        by_tool = len(edits_in_run()) - before
        applied = present = 0
        no_change = False
        if by_tool:
            error = None  # The journal, not the narrative, is the evidence.
        elif error is None:
            try:
                edits, no_change = parse_edits(raw, filename, source)
            except EditError as exc:
                edits, error = [], str(exc)
            if edits:
                applied, present, error = apply_edits(edits, filename, read_source,
                                                      patch_tool, edits_in_run)
                print(f'CREWAI TEXT EDITS {filename}: {applied} journaled, '
                      f'{present} already present', flush=True)
        if mode == 'tools' and not by_tool and runner.mode(stage.agent) == 'tools':
            runner.use_text(stage.agent)
            if runner.mode(stage.agent) == 'text':
                print(f'CREWAI TOOL MODE: the {stage.agent} model did not patch through its tools; '
                      'remaining edits use the tool-free text protocol.', flush=True)
        attempts.append({'attempt': number, 'mode': mode,
                         'result': 'failed' if error else ('no_change' if no_change else 'journal_verified'),
                         'journaled_by_tool': by_tool, 'journaled_from_text': applied,
                         'already_present': present, 'error': error})
        if error is None:
            break
        errors.append(error)
        print(f'CREWAI VALIDATION FAILURE {filename}: {error}', flush=True)
        feedback = ('PREVIOUS ATTEMPT FAILED VALIDATION: ' + error[:700] +
                    '\nThe file content above is CURRENT'
                    + (f' and already includes {applied} edit(s) from your previous answer'
                       if applied else '') +
                    '. Return only the edits that are still missing; do not repeat applied ones.')
    count = len(edits_in_run()) - file_start
    clean = bool(attempts) and attempts[-1]['error'] is None
    print(f'CREWAI VERIFIED {filename}: {count} journaled operation(s); '
          f'complete={bool(count) and clean}', flush=True)
    return {'file': filename, 'verified': bool(count), 'complete': bool(count) and clean,
            'unchanged': clean and not count, 'journaled_operations': count,
            'attempts': attempts, 'errors': errors}


# --------------------------------------------------------------------------
# Stage: explain
# --------------------------------------------------------------------------

def explain_changes(runner, tasks, cfg, request, reports, edits, test_status):
    """Have the explain stage describe the run from host-verified evidence. Returns text or None."""
    stage = tasks.get('explain')
    if stage is None:
        return None
    outcome = '\n'.join(
        f'- {r["file"]}: '
        + ('changed' if r['complete'] else
           'no change needed' if r['unchanged'] else
           'changed, but a later edit failed' if r['verified'] else 'NOT changed')
        + (f' (last error: {r["errors"][-1][:300]})' if r['errors'] and not r['complete'] and not r['unchanged'] else '')
        for r in reports) or '- no file was selected'
    print('CREWAI STAGE: explain (text) via Crew.kickoff()', flush=True)
    raw, error = runner.ask(stage.agent, 'text', stage.render(
        'text', repo=cfg.root, request=request, outcome=outcome, tests=test_status,
        diff=run_diffs(cfg, edits) or '(no changes were written)'), stage.expected_output)
    if error:
        print(f'CREWAI EXPLANATION FAILURE: {error}', flush=True)
        return None
    return raw.strip() or None


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def run_edit_pipeline(*, cfg, request, runner, tasks, patch_tool, edits_in_run, max_files,
                      pytest_tool=None, allow_tests=False, max_inline_chars=60000,
                      extra_attempts=0, explain=True):
    """Select files, edit each one, run tests if allowed, explain. Returns the status dict."""
    require_stages(tasks, ('select', 'edit'))
    selected, plan, source, _ = select_files(runner, tasks, cfg, request, max_files, for_edit=True)
    status = {'mode': 'edit', 'selected_files': selected, 'selection_source': source,
              'completed_files': [], 'incomplete_files': [], 'unchanged_files': [],
              'reports': [], 'tests': 'not run (--allow-tests disabled)', 'report': None}
    if not selected:
        status['error'] = 'No editable Python file could be identified for the request'
    for filename in selected:
        report = edit_file(runner, tasks, cfg, request, filename, plan.strip()[:PLAN_CONTEXT_CHARS],
                           patch_tool=patch_tool, edits_in_run=edits_in_run,
                           max_inline_chars=max_inline_chars, extra_attempts=extra_attempts)
        status['reports'].append(report)
        key = ('completed_files' if report['complete'] else
               'unchanged_files' if report['unchanged'] else 'incomplete_files')
        status[key].append(filename)

    if allow_tests and pytest_tool is not None and selected:
        test_targets = [f for f in selected if f.startswith('tests/') and Path(f).name.startswith('test_')]
        try:
            status['tests'] = str(pytest_tool.run(test_path=test_targets[0] if test_targets else 'tests'))[:3000]
        except Exception as exc:
            status['tests'] = f'ERROR: {exc}'
        print('TEST VERIFICATION: ' + status['tests'], flush=True)

    if explain and selected:
        status['report'] = explain_changes(runner, tasks, cfg, request, status['reports'],
                                           edits_in_run(), status['tests'])
    status.update(tool_modes=runner.tool_modes, usage=runner.usage)
    return status


def run_analysis_pipeline(*, cfg, request, runner, tasks, max_files, max_inline_chars=60000):
    """Read-only run: select relevant files, then answer the request about them.

    Nothing is written. The run succeeds when ``status['report']`` is nonempty.
    """
    require_stages(tasks, ('select', 'analyze'))
    stage = tasks['analyze']
    selected, _, source, listing = select_files(runner, tasks, cfg, request, max_files, for_edit=False)
    shown, omitted, budget = [], [], max_inline_chars
    for filename in selected:
        try:
            text = cfg.file(filename).read_bytes().decode('utf-8')
        except (ValueError, OSError, UnicodeError):
            omitted.append(filename)
            continue
        if len(text) > budget:
            omitted.append(filename)
            continue
        budget -= len(text)
        shown.append(f'<file path="{filename}">\n{text}\n</file>')
    report, error = '', None
    for mode in runner.modes_to_try(stage.agent):
        print(f'CREWAI STAGE: analyse ({mode}) via Crew.kickoff()', flush=True)
        report, error = runner.ask(stage.agent, mode, stage.render(
            mode, repo=cfg.root, request=request, listing=listing,
            file_contents='\n'.join(shown) or '(no file was selected)',
            omitted=f'Not shown (too large or unreadable): {", ".join(omitted)}' if omitted else ''),
            stage.expected_output)
        if error:
            print(f'CREWAI ANALYSIS FAILURE ({mode}): {error}', flush=True)
        if report.strip():
            break
    return {'mode': 'analysis', 'selected_files': selected, 'selection_source': source,
            'files_shown': len(shown), 'files_omitted': omitted,
            'report': report.strip() or None, 'error': error if not report.strip() else None,
            'tool_modes': runner.tool_modes, 'usage': runner.usage}

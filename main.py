import argparse
import json
import os
import shlex
import sys
import re
import traceback
import ast
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from undo import ACTIVE_RUN, get_runs, undo_run
from dotenv import load_dotenv
import yaml

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

TEXT_MODE_NOTE = (
    ' In this run you have no tools. The host application gives you the repository '
    'listing and file contents, and applies the edits you return in the requested '
    'text format. Ignore any instruction to call a tool.'
)


def extract_usage(result):
    """Return provider-reported usage from a CrewAI kickoff result, if present."""
    usage = getattr(result, 'token_usage', None)
    if usage is None:
        return None

    def number(*names):
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    prompt = number('prompt_tokens', 'input_tokens')
    completion = number('completion_tokens', 'output_tokens')
    total = number('total_tokens')
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is None:
        return None
    return {'total': total, 'prompt': prompt, 'completion': completion}


class ConsoleCapture:
    """Tee CrewAI output to the terminal and an in-memory plain-text transcript."""
    def __init__(self, stream, chunks):
        self.stream, self.chunks = stream, chunks

    def write(self, value):
        self.stream.write(value)
        self.chunks.append(value)
        return len(value)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def clean_console(value):
    """Remove terminal styling and normalize carriage-return progress updates."""
    value = ANSI.sub('', value)
    return value.replace('\r\n', '\n').replace('\r', '\n')


def run_with_log(log_path, metadata, callback):
    """Capture CrewAI's verbose transcript and append distinct task result sections."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    chunks = []
    result = None
    error = None
    sys.stdout = ConsoleCapture(old_out, chunks)
    sys.stderr = ConsoleCapture(old_err, chunks)
    try:
        result = callback()
        return result
    except Exception:
        error = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        with log_path.open('w', encoding='utf-8') as log:
            log.write('CREWAI CODING RUN\n' + '=' * 72 + '\n')
            for key, value in metadata.items():
                if key.startswith("_"):
                    continue
                log.write(f'{key}: {str(value).replace(chr(10), " ")}\n')
            log.write('\n' + '=' * 72 + '\nCREW EXECUTION TRANSCRIPT\n' + '=' * 72 + '\n')
            log.write(clean_console(''.join(chunks)).rstrip() + '\n')
            if result is not None:
                task_outputs = getattr(result, 'tasks_output', None) or []
                task_names = list(metadata.get('_task_names', []))
                if task_outputs:
                    log.write('\n' + '=' * 72 + '\nTASK RESULTS\n' + '=' * 72 + '\n')
                    for index, task_output in enumerate(task_outputs):
                        name = task_names[index] if index < len(task_names) else f'task_{index + 1}'
                        output = str(getattr(task_output, 'raw', None) or task_output)
                        log.write(f'\n--- {name} ---\n{output.rstrip()}\n')
                    if 'explain' in task_names:
                        index = task_names.index('explain')
                        if index < len(task_outputs):
                            output = str(getattr(task_outputs[index], 'raw', None) or task_outputs[index])
                            log.write('\nEXPLANATION OF WORK COMPLETED\n' + '-' * 72 + '\n' + output.rstrip() + '\n')
                log.write('\nFINAL RESULT\n' + '-' * 72 + '\n' + str(result).rstrip() + '\n')
            if error:
                log.write('\nEXECUTION ERROR\n' + '-' * 72 + '\n' + clean_console(error).rstrip() + '\n')
            log.write('\nEND OF RUN\n')

HELP_GUIDE = """CrewAI Code Agent — detailed help
=================================

SETUP
  1. Create a Python 3.12 virtual environment and install requirements.txt.
  2. Copy .env.example to .env and configure both model endpoints.
  3. Use --repo to select the LOCAL Python repository to inspect/edit.

MODEL CONFIGURATION (.env)
  REASONING_LLM_MODEL, REASONING_LLM_API_KEY, REASONING_LLM_BASE_URL
    Used by the read-only planning and explanation agents.
  CODING_LLM_MODEL, CODING_LLM_API_KEY, CODING_LLM_BASE_URL
    Used by the developer agent that edits code and may run tests.
  LLM_* variables are accepted as legacy fallbacks.
  --model overrides the coding model name only.

MODEL INDEPENDENCE
  Edits do not depend on the model calling tools. The host lists the
  repository, embeds the file being edited in the task, and applies edits the
  model returns as SEARCH/REPLACE text through the journaled patch tool.
  --tool-mode auto   Probe each model once; use tools only if they really work.
  --tool-mode text   Never rely on tool calls (any chat model).
  --tool-mode tools  Skip the probe; offer tools first, fall back to text.
  *_LLM_TEMPERATURE  Optional; a low value helps models copy code exactly.

EXAMPLES
  python main.py --repo ~/projects/sample-python-project --message "Explain this code"
  python main.py --repo ~/projects/sample-python-project -m "Fix parser errors" --allow-tests
  python main.py --repo ~/projects/sample-python-project --allowed-paths src/,docs/ --read-only-paths src/generated/,docs/archive/ -m "Update code and docs"\n  python main.py --repo ~/projects/sample-python-project --list-runs
  python main.py --repo ~/projects/sample-python-project --undo
  python main.py --repo ~/projects/sample-python-project --undo RUN_ID

OPTIONS
  --repo PATH         Target Python repository; required for runs and undo.
  -m, --message TEXT  Required for coding runs; quote messages with spaces.
  --undo [RUN_ID]     Restore latest run, or a specified run ID.
  --list-runs         Show recorded runs and their undo status.
  --backup-dir PATH   Override backups/ beside this standalone agent.
  --log-dir PATH      Override logs/ beside this standalone agent.
  --env-file PATH     Override .env beside main.py.
  --agents-config     Override config/agents.yaml.
  --tasks-config      Override config/tasks.yaml.
  --model NAME        Override coding model only.
  --tool-mode MODE    auto (default), text, or tools; see MODEL INDEPENDENCE.
  --max-inline-chars  Largest file embedded in an edit task (default 60000).
  --check-models      Test connectivity and tool calling of both models.
  -h, --help          Brief CLI usage and examples.
  --help-guide        This detailed explanation; no repo needed.

FILES AND SAFETY
  logs/       Timestamped plain-text .log execution logs.
  backups/    Original file snapshots and undo metadata.
  config/     YAML definitions for planner, developer, and explainer.
  Undo refuses unsafe overwrites when a file has changed since its run.
  Tests may execute project code: enable only for trusted repositories.
  Logs and backups may contain sensitive source code; protect both folders.
  No Git integration is required.
"""


def main():
    parser = argparse.ArgumentParser(
        description='Standalone CrewAI Python coding agent with planning, editing, explanations, backups, and undo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py --repo ~/projects/sample-python-project -m "Explain the project structure"
  python main.py --repo ~/projects/sample-python-project --message "Fix error handling" --allow-tests
  python main.py --repo ~/projects/sample-python-project --allowed-paths src/,docs/ --read-only-paths src/generated/,docs/archive/ -m "Update code and docs"\n  python main.py --repo ~/projects/sample-python-project --list-runs
  python main.py --repo ~/projects/sample-python-project --undo
  python main.py --repo ~/projects/sample-python-project --undo RUN_ID
  python main.py --help-guide

Workflow: reasoning planner -> coding developer -> reasoning explainer.
Outputs: logs/*.log; backups/ contains undo data.
Use --help-guide for setup, model configuration, safety, and all options.
"""
    )
    parser.add_argument('--help-guide', action='store_true', help='Show detailed setup and command guide, then exit (no repo required)')
    parser.add_argument('--repo', help='Local Python project directory (required except for help)')
    parser.add_argument('--undo', nargs='?', const='latest', metavar='RUN_ID', help='Undo most recent run (or specify its run ID); no LLM call')
    parser.add_argument('--list-runs', action='store_true', help='List recorded edit runs; no LLM call')
    parser.add_argument('--allowed-paths', help='Comma-separated project-relative files/directories allowed for editing; omitted means all editable files')
    parser.add_argument('--read-only-paths', help='Comma-separated project-relative files/directories that can be read but never edited')
    parser.add_argument('--backup-dir', help='Backup directory (default: standalone agent backups/; must be outside target project)')
    parser.add_argument('--allow-tests', action='store_true', help='Allow pytest execution (only in trusted/sandboxed environments)')
    parser.add_argument('--model', help='Override CODING_LLM_MODEL from .env (legacy alias)')
    parser.add_argument('--check-models', action='store_true', help='Make a small live text request to each configured model, then exit (no repo required)')
    parser.add_argument('--no-deterministic-fallback', action='store_true', help='Disable model-independent JSON patch fallback when CrewAI makes no edits')
    parser.add_argument('--max-edit-files', type=int, default=6, help='Maximum targeted files handled by focused CrewAI developer tasks')
    parser.add_argument('--legacy-monolithic', action='store_true', help='Use original single CrewAI kickoff for all tasks')
    parser.add_argument('--edit-retries', type=int, default=0, help='Extra tool-free attempts per file after the built-in ones fail validation (default: 0)')
    parser.add_argument('--tool-mode', choices=('auto', 'tools', 'text'), default='auto',
                        help='How agents reach the repository. auto (default): probe each model once and use tools only if it '
                             'really invokes them. text: never rely on tool calls; the host supplies file contents and applies '
                             'edits returned as text (works with any chat model). tools: skip the probe and offer tools first.')
    parser.add_argument('--max-inline-chars', type=int, default=60000,
                        help='Largest file (in characters) embedded in an edit task; lower it for small context windows')
    parser.add_argument('--log-dir', help='Plain-text .log directory (default: logs/ beside main.py)')
    parser.add_argument('--agents-config', help='Agents YAML path (default: config/agents.yaml)')
    parser.add_argument('--tasks-config', help='Tasks YAML path (default: config/tasks.yaml)')
    parser.add_argument('--env-file', help='Path to .env file (defaults to .env beside main.py)')
    parser.add_argument('--message', '-m', help='Required natural-language agent request for coding runs')
    args = parser.parse_args()
    if not 1 <= args.max_edit_files <= 20:
        parser.error('--max-edit-files must be between 1 and 20')
    if args.edit_retries < 0 or args.edit_retries > 5:
        parser.error('--edit-retries must be between 0 and 5')
    if args.max_inline_chars < 1000:
        parser.error('--max-inline-chars must be at least 1000')
    if args.legacy_monolithic and args.tool_mode == 'text':
        parser.error('--legacy-monolithic depends on tool calls; it cannot be combined with --tool-mode text')
    if args.help_guide:
        print(HELP_GUIDE)
        return
    if not args.repo and not args.check_models:
        parser.error('--repo is required for coding, --undo, and --list-runs')
    if args.undo is not None or args.list_runs:
        try:
            cfg = Config.create(args.repo, args.backup_dir, allowed_paths=args.allowed_paths, read_only_paths=args.read_only_paths)
            if args.list_runs:
                for entry in get_runs(cfg):
                    status = 'undone' if entry.get('undone') else 'available'
                    print(f"{entry['run_id']}  {status}  {len(entry['edits'])} edit(s)")
                return
            result = undo_run(cfg, args.undo)
            print(f"Restored {len(result['restored_files'])} file(s) from run {result['run_id']}")
            for filename in result['restored_files']:
                print(f'  {filename}')
            return
        except (ValueError, OSError) as exc:
            parser.error(f'Undo failed: {exc}')
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_file, override=False)
    # Independent model configuration; no custom token/output limits.
    def model_config(prefix, override=None):
        model_name = override or os.getenv(f'{prefix}_LLM_MODEL') or os.getenv('LLM_MODEL', 'openai/gpt-4o')
        url = (os.getenv(f'{prefix}_LLM_BASE_URL') or os.getenv('LLM_BASE_URL') or '').strip() or None
        key = (os.getenv(f'{prefix}_LLM_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or '').strip()
        raw_context = (os.getenv(f'{prefix}_LLM_CONTEXT_WINDOW') or '').strip()
        context_window = None
        if raw_context:
            try:
                context_window = int(raw_context)
            except ValueError:
                parser.error(f'{prefix}_LLM_CONTEXT_WINDOW must be a positive integer')
            if context_window <= 0:
                parser.error(f'{prefix}_LLM_CONTEXT_WINDOW must be a positive integer')
        raw_body = (os.getenv(f'{prefix}_LLM_EXTRA_BODY') or '').strip()
        extra_body = None
        if raw_body:
            try:
                extra_body = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                parser.error(f'{prefix}_LLM_EXTRA_BODY must be valid JSON: {exc}')
            if not isinstance(extra_body, dict):
                parser.error(f'{prefix}_LLM_EXTRA_BODY must be a JSON object')
        raw_temperature = (os.getenv(f'{prefix}_LLM_TEMPERATURE') or '').strip()
        temperature = None
        if raw_temperature:
            try:
                temperature = float(raw_temperature)
            except ValueError:
                parser.error(f'{prefix}_LLM_TEMPERATURE must be a number')
        return {'model': model_name, 'base_url': url, 'api_key': key,
                'context_window': context_window, 'extra_body': extra_body,
                'temperature': temperature}

    coding_config = model_config('CODING', args.model)
    reasoning_config = model_config('REASONING')
    request = (args.message or '').strip()
    if not request and not args.check_models:
        parser.error('A coding request is required: pass --message "Describe the change" (or -m)')
    for label, settings in [('coding', coding_config), ('reasoning', reasoning_config)]:
        if not settings['api_key']:
            parser.error(f'Missing {label} API key: set {label.upper()}_LLM_API_KEY in .env')
    if not args.check_models:
        try:
            cfg = Config.create(args.repo, args.backup_dir, allowed_paths=args.allowed_paths, read_only_paths=args.read_only_paths)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool as crew_tool

    from agent_tools import make_tools
    from tool_probe import probe_tool_calling

    model_settings = {'coding': coding_config, 'reasoning': reasoning_config}
    probe_cache = {}

    def probe(model_role):
        """Probe a model once per endpoint; both roles often share one model."""
        settings = model_settings[model_role]
        key = (settings['model'], settings['base_url'])
        if key not in probe_cache:
            probe_cache[key] = probe_tool_calling(
                llms[model_role], Agent=Agent, Task=Task, Crew=Crew, Process=Process, tool=crew_tool)
        return probe_cache[key]

    def create_llm(settings):
        kwargs = {'model': settings['model'], 'api_key': settings['api_key']}
        if settings['base_url']:
            kwargs['base_url'] = settings['base_url']
        if settings['context_window'] is not None:
            kwargs['context_window_size'] = settings['context_window']
        if settings['extra_body'] is not None:
            kwargs['extra_body'] = settings['extra_body']
        if settings['temperature'] is not None:
            kwargs['temperature'] = settings['temperature']
        return LLM(**kwargs)

    llms = {'coding': create_llm(coding_config), 'reasoning': create_llm(reasoning_config)}
    if args.check_models:
        failed = False
        for label, llm in llms.items():
            try:
                result = llm.call([{'role': 'user', 'content': 'Reply with exactly: OK'}])
                print(f'{label} model responded: {str(result)[:200]}')
            except Exception as exc:
                failed = True
                print(f'{label} model failed ({type(exc).__name__}): {exc}', file=sys.stderr)
        if failed:
            raise SystemExit(1)
        print('Both models responded. Checking CrewAI tool calling (one more small request per model)...')
        for label in llms:
            supported, detail = probe(label)
            print(f'{label} model tool calling: {"works" if supported else "NOT usable"} ({detail}); '
                  f'--tool-mode auto will use {"tools" if supported else "the tool-free text protocol"}.')
        print('Edits work in either case; tool calling only decides how the agents reach the repository.')
        return
    tool_events = []
    def capture_tool_event(event):
        event = {'timestamp': datetime.now(timezone.utc).isoformat(), **event}
        tool_events.append(event)
        # Console output is captured by run_with_log and stored in the .log file.
        print('FILE_TOOL_EVENT ' + json.dumps(event, ensure_ascii=False), flush=True)

    tools = make_tools(cfg, allow_tests=args.allow_tests, on_event=capture_tool_event)
    tool_lookup = {tool.name: tool for tool in tools}
    base = Path(__file__).resolve().parent
    agents_path = Path(args.agents_config).expanduser().resolve() if args.agents_config else base / 'config' / 'agents.yaml'
    tasks_path = Path(args.tasks_config).expanduser().resolve() if args.tasks_config else base / 'config' / 'tasks.yaml'

    def load_mapping(path, section):
        with path.open(encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict) or not isinstance(data.get(section), dict) or not data[section]:
            raise ValueError(f'{path}: expected nonempty mapping under {section!r}')
        return data[section]

    def render(value):
        if not isinstance(value, str):
            raise ValueError('Agent/task text fields must be strings')
        for token, replacement in {
            '{repo}': str(cfg.root),
            '{request}': request,
            '{test_instruction}': (
                'Run pytest on relevant tests after edits.' if args.allow_tests
                else 'Tests are disabled; report that tests were not run.'
            ),
        }.items():
            value = value.replace(token, replacement)
        return value

    try:
        agent_defs = load_mapping(agents_path, 'agents')
        task_defs = load_mapping(tasks_path, 'tasks')
        agents = {}
        text_agents = {}
        for name, spec in agent_defs.items():
            if not isinstance(spec, dict):
                raise ValueError(f'Agent {name} must be a mapping')
            selected = spec.get('tools', [])
            if not isinstance(selected, list) or any(not isinstance(t, str) for t in selected):
                raise ValueError(f'Agent {name}: tools must be a list of names')
            unknown = set(selected) - set(tool_lookup)
            if unknown:
                raise ValueError(f'Agent {name}: unknown tools: {sorted(unknown)}')
            model_role = spec.get('model', 'coding')
            if model_role not in llms:
                raise ValueError(f'Agent {name}: model must be coding or reasoning')
            def build(with_tools, spec=spec, selected=selected, model_role=model_role):
                options = {}
                if 'use_system_prompt' in spec:
                    # Some models reject the system role; let the YAML turn it off.
                    options['use_system_prompt'] = bool(spec['use_system_prompt'])
                return Agent(
                    role=render(spec['role']),
                    goal=render(spec['goal']),
                    backstory=render(spec['backstory']) + ('' if with_tools else TEXT_MODE_NOTE),
                    tools=[tool_lookup[t] for t in selected] if with_tools else [],
                    llm=llms[model_role],
                    verbose=bool(spec.get('verbose', True)),
                    allow_delegation=False,
                    max_iter=int(spec.get('max_iter', 20)),
                    **options,
                )
            agents[name] = build(True)
            # Same agent without tools: used when a model cannot call them reliably.
            text_agents[name] = build(False)
        tasks = []
        for name, spec in task_defs.items():
            if not isinstance(spec, dict) or spec.get('agent') not in agents:
                raise ValueError(f'Task {name}: unknown or missing agent')
            tasks.append(Task(
                description=render(spec['description']),
                expected_output=render(spec['expected_output']),
                agent=agents[spec['agent']],
                context=list(tasks) if spec.get('use_previous_context', True) else [],
            ))
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Invalid YAML configuration: {exc}')

    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else Path(__file__).resolve().parent / 'logs'
    if log_dir == cfg.root or log_dir.is_relative_to(cfg.root):
        parser.error('Log directory must be outside the target repository')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = log_dir / f'run-{stamp}.log'

    usage_result = [None]
    verified_edits = [0]
    attempts = [0]

    from focused_crew import run_focused_crew

    def edits_in_run():
        entry = next((e for e in get_runs(cfg)
                      if e['run_id'] == stamp and not e.get('undone')), None)
        return entry['edits'] if entry else []

    execution_status = {}

    def resolve_tool_modes():
        """Decide per agent whether tool calls can be relied on for this run."""
        modes = {}
        for name in ('planner', 'developer'):
            spec = agent_defs.get(name)
            if not isinstance(spec, dict):
                continue
            if args.tool_mode != 'auto':
                modes[name] = args.tool_mode
                print(f'TOOL MODE {name}: {modes[name]} (--tool-mode {args.tool_mode})')
            elif not spec.get('tools'):
                modes[name] = 'text'
                print(f'TOOL MODE {name}: text (no tools configured)')
            else:
                supported, detail = probe(spec.get('model', 'coding'))
                modes[name] = 'tools' if supported else 'text'
                print(f'TOOL MODE {name}: {modes[name]} (probe: {detail})')
        return modes

    def execute():
        print(f'Project: {cfg.root}\nBackups: {cfg.backup_root}\nCoding model: {coding_config["model"]}\nReasoning model: {reasoning_config["model"]}\nPytest enabled: {args.allow_tests}\n')
        if args.legacy_monolithic:
            result = Crew(agents=list(agents.values()), tasks=tasks,
                          process=Process.sequential, verbose=True).kickoff()
            usage_result[0] = extract_usage(result)
            verified_edits[0] = len(edits_in_run())
            execution_status.update({'mode': 'legacy', 'verified_edits': verified_edits[0]})
        else:
            result, status = run_focused_crew(
                cfg=cfg, request=request, agents=agents, task_defs=task_defs,
                render=render, tool_lookup=tool_lookup, max_files=args.max_edit_files,
                allow_tests=args.allow_tests, edits_in_run=edits_in_run,
                Crew=Crew, Task=Task, Process=Process,
                text_agents=text_agents, tool_modes=resolve_tool_modes(),
                max_inline_chars=args.max_inline_chars, extra_attempts=args.edit_retries,
            )
            execution_status.update(status)
            verified_edits[0] = len(edits_in_run())
            usage_result[0] = status.get('usage')
        print('\n=== VERIFIED EXECUTION STATUS ===')
        print(json.dumps({k: v for k, v in execution_status.items() if k != 'usage'}, indent=2))
        print(f'Journaled edits: {verified_edits[0]}')
        if not verified_edits[0]:
            print('WARNING: No files were changed. Model output is NOT proof of an edit.')
        if execution_status.get('incomplete_files'):
            print('WARNING: Incomplete files: ' + ', '.join(execution_status['incomplete_files']))
        return result

    token = ACTIVE_RUN.set(stamp)
    try:
        run_with_log(log_path, {
            'Timestamp (UTC)': stamp,
            'Project': cfg.root,
            'Request': request,
            'Coding model': coding_config['model'],
            'Reasoning model': reasoning_config['model'],
            'Pytest enabled': args.allow_tests,
            'Tool mode': args.tool_mode,
            'Agents config': agents_path,
            'Tasks config': tasks_path,
            '_task_names': list(task_defs) if args.legacy_monolithic else [],
        }, execute)
    finally:
        ACTIVE_RUN.reset(token)
        # Match the exact execution ID; avoid accidentally showing a different run.
        run_entry = next((entry for entry in get_runs(cfg)
                          if entry['run_id'] == stamp and not entry.get('undone')), None)
        if run_entry:
            undo_command = (
                f'python main.py --repo {shlex.quote(str(cfg.root))} '
                f'--undo {shlex.quote(stamp)}'
            )
            if args.backup_dir:
                undo_command += f' --backup-dir {shlex.quote(str(cfg.backup_root))}'
            print(f'\nUndo this run ({len(run_entry["edits"])} edit(s)):\n{undo_command}')
            if log_path.is_file():
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\nUNDO THIS RUN\n' + '-' * 72 + '\n')
                    log.write(undo_command + '\n')
        else:
            print('\nUndo this run: unavailable (no recorded file edits).')
            if log_path.is_file():
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\nUNDO THIS RUN\n' + '-' * 72 + '\n')
                    log.write('No recorded file edits; nothing to undo.\n')
        if usage_result[0] is not None:
            usage = usage_result[0]
            usage_lines = [f'Total tokens: {usage["total"]:,}']
            if usage['prompt'] is not None:
                usage_lines.append(f'Prompt tokens: {usage["prompt"]:,}')
            if usage['completion'] is not None:
                usage_lines.append(f'Completion tokens: {usage["completion"]:,}')
            print('\nProvider-reported token usage (CrewAI):')
            for line in usage_lines:
                print(line)
            if log_path.is_file():
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\nPROVIDER-REPORTED TOKEN USAGE (CREWAI)\n' + '-' * 72 + '\n' + '\n'.join(usage_lines) + '\n')
        if log_path.is_file():
            with log_path.open('a', encoding='utf-8') as log:
                log.write('\nFILE EDIT VERIFICATION\n' + '-' * 72 + '\n')
                log.write(f'Journaled edits: {len(run_entry["edits"]) if run_entry else 0}\n')
                log.write(f'Developer retry attempts: {attempts[0]}\n')
                log.write('VERIFIED STATUS\n' + json.dumps({k: v for k, v in execution_status.items() if k != 'usage'}, indent=2) + '\n')
                log.write(f'Tool events: {len(tool_events)}\n')
                for event in tool_events:
                    log.write(json.dumps(event, ensure_ascii=False) + '\n')
                if not run_entry:
                    log.write('WARNING: No files were changed.\n')
        print(f'\nLog file: {log_path}')
        if not run_entry:
            raise SystemExit(2)


if __name__ == '__main__':
    main()

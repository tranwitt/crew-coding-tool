import argparse
import os
import shlex
import sys
import re
import traceback
from html import escape
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from undo import ACTIVE_RUN, get_runs, undo_run
from dotenv import load_dotenv
import yaml

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


def fence(value):
    """Use a fence longer than any backtick sequence in tool/model output."""
    longest = max((len(x) for x in re.findall(r'`+', value)), default=0)
    marker = '`' * max(4, longest + 1)
    return f'{marker}text\n{value.rstrip()}\n{marker}\n'


def write_html_log(markdown_path):
    """Render our structured run-log format using only the Python standard library."""
    source = markdown_path.read_text(encoding='utf-8')
    lines = source.splitlines()
    sections = []
    in_fence = False
    fence_marker = ''
    for line in lines:
        stripped = line.strip()
        if not in_fence and re.match(r'^`{4,}text$', stripped):
            in_fence = True
            fence_marker = stripped[:-4]
            sections.append(('code_start', ''))
        elif in_fence and stripped == fence_marker:
            in_fence = False
            sections.append(('code_end', ''))
        elif in_fence:
            sections.append(('code', line))
        elif line.startswith('### '):
            sections.append(('h3', line[4:]))
        elif line.startswith('## '):
            sections.append(('h2', line[3:]))
        elif line.startswith('# '):
            sections.append(('h1', line[2:]))
        elif line.startswith('- **') and ':** ' in line:
            key, value = line[4:].split(':** ', 1)
            sections.append(('meta', (key, value)))
        elif stripped:
            sections.append(('p', line))
    parts = []
    code_lines = []
    for kind, value in sections:
        if kind == 'code_start':
            code_lines = []
        elif kind == 'code':
            code_lines.append(value)
        elif kind == 'code_end':
            parts.append('<pre><code>' + escape('\n'.join(code_lines)) + '</code></pre>')
        elif kind == 'meta':
            parts.append('<div class="meta"><strong>' + escape(value[0]) + ':</strong> ' + escape(value[1]) + '</div>')
        else:
            parts.append(f'<{kind}>' + escape(value) + f'</{kind}>')
    title = escape(markdown_path.stem)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>{title} — CrewAI Log</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55; }}
h1, h2 {{ border-bottom: 1px solid #8885; padding-bottom: .4rem; }}
h2 {{ margin-top: 2rem; }} h3 {{ margin-top: 1.5rem; }}
pre {{ overflow-x: auto; padding: 1rem; border: 1px solid #8885; border-radius: 8px;
 background: #8881; white-space: pre-wrap; overflow-wrap: anywhere; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
.meta {{ margin: .25rem 0; overflow-wrap: anywhere; }}
</style></head><body>{''.join(parts)}</body></html>"""
    output = markdown_path.with_suffix('.html')
    output.write_text(page, encoding='utf-8')
    return output


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
            log.write('# CrewAI Coding Run\n\n')
            for key, value in metadata.items():
                if key.startswith("_"):
                    continue
                log.write(f'- **{key}:** {str(value).replace(chr(10), " ")}\n')
            log.write('\n## Crew execution transcript\n\n')
            log.write(fence(clean_console(''.join(chunks))))
            if result is not None:
                task_outputs = getattr(result, 'tasks_output', None) or []
                task_names = list(metadata.get('_task_names', []))
                if task_outputs:
                    log.write('\n## Task results\n\n')
                    for index, task_output in enumerate(task_outputs):
                        name = task_names[index] if index < len(task_names) else f'task_{index + 1}'
                        output = str(getattr(task_output, 'raw', None) or task_output)
                        log.write(f'### {name}\n\n' + fence(output) + '\n')
                    if 'explain' in task_names:
                        index = task_names.index('explain')
                        if index < len(task_outputs):
                            output = str(getattr(task_outputs[index], 'raw', None) or task_outputs[index])
                            log.write('## Explanation of work completed\n\n' + fence(output) + '\n')
                log.write('\n## Final result\n\n' + fence(str(result)) + '\n')
            if error:
                log.write('\n## Execution error\n\n' + fence(clean_console(error)) + '\n')
            log.write('\n_End of run._\n')

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
  -h, --help          Brief CLI usage and examples.
  --help-guide        This detailed explanation; no repo needed.

FILES AND SAFETY
  logs/       Timestamped Markdown and HTML execution logs.
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
Outputs: logs/*.md and logs/*.html; backups/ contains undo data.
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
    parser.add_argument('--log-dir', help='Markdown and HTML log directory (default: logs/ beside main.py)')
    parser.add_argument('--agents-config', help='Agents YAML path (default: config/agents.yaml)')
    parser.add_argument('--tasks-config', help='Tasks YAML path (default: config/tasks.yaml)')
    parser.add_argument('--env-file', help='Path to .env file (defaults to .env beside main.py)')
    parser.add_argument('--message', '-m', help='Required natural-language agent request for coding runs')
    args = parser.parse_args()
    if args.help_guide:
        print(HELP_GUIDE)
        return
    if not args.repo:
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
    # Independent LLM configurations; legacy LLM_* variables remain a fallback.
    def model_config(prefix, override=None):
        model_name = override or os.getenv(f'{prefix}_LLM_MODEL') or os.getenv('LLM_MODEL', 'openai/gpt-4o')
        url = (os.getenv(f'{prefix}_LLM_BASE_URL') or os.getenv('LLM_BASE_URL') or '').strip() or None
        key = (os.getenv(f'{prefix}_LLM_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or '').strip()
        return {'model': model_name, 'base_url': url, 'api_key': key}

    coding_config = model_config('CODING', args.model)
    reasoning_config = model_config('REASONING')
    request = (args.message or '').strip()
    if not request:
        parser.error('A coding request is required: pass --message "Describe the change" (or -m)')
    for label, settings in [('coding', coding_config), ('reasoning', reasoning_config)]:
        if not settings['api_key']:
            parser.error(f'Missing {label} API key: set {label.upper()}_LLM_API_KEY in .env')
    try:
        cfg = Config.create(args.repo, args.backup_dir, allowed_paths=args.allowed_paths, read_only_paths=args.read_only_paths)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    from crewai import Agent, Task, Crew, Process, LLM

    from agent_tools import make_tools

    def create_llm(settings):
        kwargs = {'model': settings['model'], 'api_key': settings['api_key']}
        if settings['base_url']:
            kwargs['base_url'] = settings['base_url']
        return LLM(**kwargs)

    llms = {'coding': create_llm(coding_config), 'reasoning': create_llm(reasoning_config)}
    tools = make_tools(cfg, allow_tests=args.allow_tests)
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
            agents[name] = Agent(
                role=render(spec['role']),
                goal=render(spec['goal']),
                backstory=render(spec['backstory']),
                tools=[tool_lookup[t] for t in selected],
                llm=llms[model_role],
                verbose=bool(spec.get('verbose', True)),
                allow_delegation=False,
                max_iter=int(spec.get('max_iter', 20)),
            )
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
    log_path = log_dir / f'run-{stamp}.md'

    usage_result = [None]

    def execute():
        print(f'Project: {cfg.root}\nBackups: {cfg.backup_root}\nCoding model: {coding_config["model"]}\nReasoning model: {reasoning_config["model"]}\nPytest enabled: {args.allow_tests}\n')
        result = Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential, verbose=True).kickoff()
        print('\n=== RESULT ===\n', result)
        usage_result[0] = extract_usage(result)
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
            'Agents config': agents_path,
            'Tasks config': tasks_path,
            '_task_names': list(task_defs),
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
                    log.write('\n## Undo this run\n\n')
                    log.write(fence(undo_command) + '\n')
        else:
            print('\nUndo this run: unavailable (no recorded file edits).')
            if log_path.is_file():
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\n## Undo this run\n\n')
                    log.write('No recorded file edits; nothing to undo.\n')
        if usage_result[0] is not None:
            usage = usage_result[0]
            usage_lines = ['## Provider-reported token usage (CrewAI)', '',
                           f'- **Total tokens:** {usage["total"]:,}']
            if usage['prompt'] is not None:
                usage_lines.append(f'- **Prompt tokens:** {usage["prompt"]:,}')
            if usage['completion'] is not None:
                usage_lines.append(f'- **Completion tokens:** {usage["completion"]:,}')
            print('\nProvider-reported token usage (CrewAI):')
            for line in usage_lines[2:]:
                print(line.removeprefix('- ').replace('**', ''))
            if log_path.is_file():
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\n'.join(usage_lines) + '\n')
        print(f'\nMarkdown log: {log_path}')
        if log_path.is_file():
            try:
                html_path = write_html_log(log_path)
                print(f'HTML log: {html_path}')
            except Exception as exc:
                print(f'WARNING: Could not generate HTML log: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()

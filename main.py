"""Command line entry point: parse options, build the crew, run one pipeline, report."""
import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from config import Config
from models import ROLES, check_models, create_llm, load_model_settings
from run_log import append_section, capture_run
from undo import ACTIVE_RUN, get_runs, undo_run

HERE = Path(__file__).resolve().parent

EXAMPLES = """  python main.py --repo ~/projects/sample-python-project -m "Fix error handling" --allow-tests
  python main.py --repo ~/projects/sample-python-project --analyze -m "Explain the project structure"
  python main.py --repo ~/projects/sample-python-project --allowed-paths src/ --read-only-paths src/generated/ -m "Update the parser"
  python main.py --repo ~/projects/sample-python-project --list-runs
  python main.py --repo ~/projects/sample-python-project --undo
  python main.py --repo ~/projects/sample-python-project --undo RUN_ID
"""

HELP_GUIDE = f"""CrewAI Code Agent - detailed help
=================================

SETUP
  1. Create a Python 3.12 virtual environment and install requirements.txt.
  2. Copy .env.example to .env and configure both model endpoints.
  3. Use --repo to select the LOCAL Python repository to inspect/edit.

RUN TYPES
  Edit run (default)   select files -> edit each file -> run tests (with
                       --allow-tests) -> explain. Succeeds (exit 0) when at
                       least one edit was written and journaled; exit 2 when
                       nothing was changed.
  Analysis run         --analyze: select files -> answer the request. Never
                       writes to the repository. Succeeds (exit 0) when a
                       report was produced; exit 1 otherwise.

MODEL CONFIGURATION (.env)
  REASONING_LLM_MODEL, REASONING_LLM_API_KEY, REASONING_LLM_BASE_URL
    Used by the planner and explainer agents.
  CODING_LLM_MODEL, CODING_LLM_API_KEY, CODING_LLM_BASE_URL
    Used by the developer agent.
  LLM_* variables are accepted as legacy fallbacks.
  --model overrides the coding model name only.

MODEL INDEPENDENCE
  Nothing depends on the model calling tools. The host lists the repository,
  embeds file contents in each task, and applies edits the model returns as
  SEARCH/REPLACE text through the journaled patch tool.
  --tool-mode auto   Probe each model once; use tools only if they really work.
  --tool-mode text   Never rely on tool calls (any chat model).
  --tool-mode tools  Skip the probe; offer tools first, fall back to text.
  *_LLM_TEMPERATURE  Optional; a low value helps models copy code exactly.

EXAMPLES
{EXAMPLES}
OPTIONS
  --repo PATH         Target Python repository; required for runs and undo.
  -m, --message TEXT  The request; quote messages with spaces.
  --analyze           Read-only analysis run; see RUN TYPES.
  --allow-tests       Run pytest after edits (trusted repositories only).
  --no-explain        Skip the explanation stage of an edit run (one model call).
  --max-edit-files N  Most files handled in one run (default 6).
  --edit-retries N    Extra attempts per file after the built-in ones.
  --tool-mode MODE    auto (default), text, or tools; see MODEL INDEPENDENCE.
  --max-inline-chars  Largest file embedded in a task (default 60000).
  --check-models      Test connectivity and tool calling of both models.
  --undo [RUN_ID]     Restore latest run, or a specified run ID.
  --list-runs         Show recorded runs and their undo status.
  --backup-dir PATH   Override backups/ beside this standalone agent.
  --log-dir PATH      Override logs/ beside this standalone agent.
  --env-file PATH     Override .env beside main.py.
  --agents-config     Override config/agents.yaml.
  --model NAME        Override coding model only.
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


def build_parser():
    parser = argparse.ArgumentParser(
        description='Standalone CrewAI Python coding agent: analysis, editing, explanations, backups, and undo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Examples:\n{EXAMPLES}  python main.py --help-guide\n\n'
               'Outputs: logs/*.log; backups/ contains undo data.\n'
               'Use --help-guide for setup, run types, model configuration, and all options.\n')
    add = parser.add_argument
    add('--help-guide', action='store_true', help='Show detailed setup and command guide, then exit (no repo required)')
    add('--repo', help='Local Python project directory (required except for help and --check-models)')
    add('--message', '-m', help='Natural-language request')
    add('--analyze', action='store_true', help='Read-only run: analyse the repository and answer the request; never edits files')
    add('--undo', nargs='?', const='latest', metavar='RUN_ID', help='Undo most recent run (or specify its run ID); no LLM call')
    add('--list-runs', action='store_true', help='List recorded edit runs; no LLM call')
    add('--allowed-paths', help='Comma-separated project-relative files/directories allowed for editing; omitted means all editable files')
    add('--read-only-paths', help='Comma-separated project-relative files/directories that can be read but never edited')
    add('--backup-dir', help='Backup directory (default: standalone agent backups/; must be outside target project)')
    add('--allow-tests', action='store_true', help='Allow pytest execution (only in trusted/sandboxed environments)')
    add('--no-explain', action='store_true', help='Skip the explanation stage after an edit run')
    add('--model', help='Override CODING_LLM_MODEL from .env')
    add('--check-models', action='store_true', help='Test connectivity and tool calling of each configured model, then exit (no repo required)')
    add('--max-edit-files', type=int, default=6, help='Maximum files handled in one run')
    add('--edit-retries', type=int, default=0, help='Extra tool-free attempts per file after the built-in ones fail validation (default: 0)')
    add('--tool-mode', choices=('auto', 'tools', 'text'), default='auto',
        help='How agents reach the repository. auto (default): probe each model once and use tools only if it '
             'really invokes them. text: never rely on tool calls; the host supplies file contents and applies '
             'edits returned as text (works with any chat model). tools: skip the probe and offer tools first.')
    add('--max-inline-chars', type=int, default=60000,
        help='Largest file (in characters) embedded in a task; lower it for small context windows')
    add('--log-dir', help='Plain-text .log directory (default: logs/ beside main.py)')
    add('--agents-config', help='Agents YAML path (default: config/agents.yaml)')
    add('--env-file', help='Path to .env file (defaults to .env beside main.py)')
    return parser


def run_undo(args, parser):
    """--list-runs and --undo: journal operations that never call a model."""
    try:
        cfg = Config.create(args.repo, args.backup_dir, allowed_paths=args.allowed_paths,
                            read_only_paths=args.read_only_paths)
        if args.list_runs:
            for entry in get_runs(cfg):
                status = 'undone' if entry.get('undone') else 'available'
                print(f"{entry['run_id']}  {status}  {len(entry['edits'])} edit(s)")
            return
        result = undo_run(cfg, args.undo)
        print(f"Restored {len(result['restored_files'])} file(s) from run {result['run_id']}")
        for filename in result['restored_files']:
            print(f'  {filename}')
    except (ValueError, OSError) as exc:
        parser.error(f'Undo failed: {exc}')


def report_run(*, args, cfg, stamp, log_path, status, tool_events):
    """Print and log the verified outcome of a run. Returns the process exit code."""
    summary = {k: v for k, v in status.items() if k not in ('usage', 'report')}
    report = status.get('report')
    run_entry = next((entry for entry in get_runs(cfg)
                      if entry['run_id'] == stamp and not entry.get('undone')), None)
    edits = run_entry['edits'] if run_entry else []

    if report:
        title = 'ANALYSIS REPORT' if args.analyze else 'EXPLANATION OF WORK COMPLETED'
        print(f'\n=== {title} ===\n{report}')
        append_section(log_path, title, report)

    if run_entry:
        undo_command = f'python main.py --repo {shlex.quote(str(cfg.root))} --undo {shlex.quote(stamp)}'
        if args.backup_dir:
            undo_command += f' --backup-dir {shlex.quote(str(cfg.backup_root))}'
        print(f'\nUndo this run ({len(edits)} edit(s)):\n{undo_command}')
        append_section(log_path, 'UNDO THIS RUN', undo_command)
    elif not args.analyze:
        print('\nUndo this run: unavailable (no recorded file edits).')
        append_section(log_path, 'UNDO THIS RUN', 'No recorded file edits; nothing to undo.')

    usage = status.get('usage') or {}
    if usage.get('total'):
        lines = [f'Total tokens: {usage["total"]:,}', f'Prompt tokens: {usage.get("prompt", 0):,}',
                 f'Completion tokens: {usage.get("completion", 0):,}']
        print('\nProvider-reported token usage (CrewAI):\n' + '\n'.join(lines))
        append_section(log_path, 'PROVIDER-REPORTED TOKEN USAGE (CREWAI)', '\n'.join(lines))

    verification = [f'Journaled edits: {len(edits)}', 'VERIFIED STATUS', json.dumps(summary, indent=2),
                    f'Tool events: {len(tool_events)}']
    verification += [json.dumps(event, ensure_ascii=False) for event in tool_events]
    if not edits and not args.analyze:
        verification.append('WARNING: No files were changed.')
    append_section(log_path, 'RUN VERIFICATION', '\n'.join(verification))
    append_section(log_path, 'END OF RUN', '')
    print(f'\nLog file: {log_path}')

    if args.analyze:
        return 0 if report else 1
    return 0 if edits else 2


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.max_edit_files <= 20:
        parser.error('--max-edit-files must be between 1 and 20')
    if not 0 <= args.edit_retries <= 5:
        parser.error('--edit-retries must be between 0 and 5')
    if args.max_inline_chars < 1000:
        parser.error('--max-inline-chars must be at least 1000')
    if args.help_guide:
        print(HELP_GUIDE)
        return
    if not args.repo and not args.check_models:
        parser.error('--repo is required for runs, --undo, and --list-runs')
    if args.undo is not None or args.list_runs:
        return run_undo(args, parser)

    request = (args.message or '').strip()
    if not request and not args.check_models:
        parser.error('A request is required: pass --message "Describe the change or question" (or -m)')
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else HERE / '.env'
    load_dotenv(dotenv_path=env_file, override=False)
    try:
        settings = load_model_settings(args.model)
        cfg = None if args.check_models else Config.create(
            args.repo, args.backup_dir, allowed_paths=args.allowed_paths, read_only_paths=args.read_only_paths)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    # CrewAI is imported late so help, undo and argument errors stay instant.
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool as crew_tool

    from agent_tools import make_tools
    from crew_setup import build_agents, load_agent_specs, make_probe, resolve_tool_modes
    from pipeline import CrewRunner, run_analysis_pipeline, run_edit_pipeline

    llms = {role: create_llm(settings[role], LLM) for role in ROLES}
    probe = make_probe(llms, settings, Agent=Agent, Task=Task, Crew=Crew, Process=Process, tool=crew_tool)
    if args.check_models:
        raise SystemExit(0 if check_models(llms, probe) else 1)

    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else HERE / 'logs'
    if log_dir == cfg.root or log_dir.is_relative_to(cfg.root):
        parser.error('Log directory must be outside the target repository')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = log_dir / f'run-{stamp}.log'

    tool_events = []

    def capture_tool_event(event):
        event = {'timestamp': datetime.now(timezone.utc).isoformat(), **event}
        tool_events.append(event)
        # Console output is captured by capture_run and stored in the .log file.
        print('FILE_TOOL_EVENT ' + json.dumps(event, ensure_ascii=False), flush=True)

    tool_lookup = {tool.name: tool for tool in
                   make_tools(cfg, allow_tests=args.allow_tests, on_event=capture_tool_event)}
    agents_path = Path(args.agents_config).expanduser().resolve() if args.agents_config else HERE / 'config' / 'agents.yaml'
    try:
        specs = load_agent_specs(agents_path)
        agents, text_agents = build_agents(
            specs, llms, tool_lookup, Agent,
            variables={'{repo}': str(cfg.root), '{request}': request},
            # An analysis run must not be able to write, whatever the YAML grants.
            withhold=('patch_project_file',) if args.analyze else ())
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Invalid agents configuration: {exc}')

    def edits_in_run():
        entry = next((e for e in get_runs(cfg) if e['run_id'] == stamp and not e.get('undone')), None)
        return entry['edits'] if entry else []

    status = {}

    def execute():
        print(f'Project: {cfg.root}\nBackups: {cfg.backup_root}\n'
              f'Run type: {"analysis (read-only)" if args.analyze else "edit"}\n'
              f'Coding model: {settings["coding"]["model"]}\nReasoning model: {settings["reasoning"]["model"]}\n'
              f'Pytest enabled: {args.allow_tests}\n')
        runner = CrewRunner(agents=agents, text_agents=text_agents,
                            tool_modes=resolve_tool_modes(specs, args.tool_mode, probe),
                            Crew=Crew, Task=Task, Process=Process)
        if args.analyze:
            status.update(run_analysis_pipeline(
                cfg=cfg, request=request, runner=runner, max_files=args.max_edit_files,
                max_inline_chars=args.max_inline_chars))
        else:
            status.update(run_edit_pipeline(
                cfg=cfg, request=request, runner=runner,
                patch_tool=tool_lookup['patch_project_file'], pytest_tool=tool_lookup['run_pytest'],
                edits_in_run=edits_in_run, max_files=args.max_edit_files,
                allow_tests=args.allow_tests, max_inline_chars=args.max_inline_chars,
                extra_attempts=args.edit_retries, explain=not args.no_explain))
        print('\n=== VERIFIED EXECUTION STATUS ===')
        print(json.dumps({k: v for k, v in status.items() if k not in ('usage', 'report')}, indent=2))
        if not args.analyze:
            print(f'Journaled edits: {len(edits_in_run())}')
            if not edits_in_run():
                print('WARNING: No files were changed. Model output is NOT proof of an edit.')
            if status.get('incomplete_files'):
                print('WARNING: Incomplete files: ' + ', '.join(status['incomplete_files']))

    token = ACTIVE_RUN.set(stamp)
    try:
        capture_run(log_path, {
            'Timestamp (UTC)': stamp,
            'Project': cfg.root,
            'Request': request,
            'Run type': 'analysis' if args.analyze else 'edit',
            'Coding model': settings['coding']['model'],
            'Reasoning model': settings['reasoning']['model'],
            'Pytest enabled': args.allow_tests,
            'Tool mode': args.tool_mode,
            'Agents config': agents_path,
        }, execute)
    finally:
        ACTIVE_RUN.reset(token)
        # Report even when the run raised, so written edits always get an undo command.
        exit_code = report_run(args=args, cfg=cfg, stamp=stamp, log_path=log_path,
                               status=status, tool_events=tool_events)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()

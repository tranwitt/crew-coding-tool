# Standalone CrewAI Python Code Agent

Edit an existing local Python project using CrewAI. No Git required. The agent saves a backup before each edit and supports safe local undo and Markdown and HTML logs for each run. Agents and tasks are defined in YAML. A read-only reasoning planner prepares a change plan, the coding agent edits files, and a read-only reasoning explainer summarizes the results.

## Linux setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to set two independent configurations:

```dotenv
REASONING_LLM_BASE_URL=https://api.openai.com/v1
REASONING_LLM_API_KEY=your-reasoning-key
REASONING_LLM_MODEL=openai/gpt-4o

CODING_LLM_BASE_URL=https://api.openai.com/v1
CODING_LLM_API_KEY=your-coding-key
CODING_LLM_MODEL=openai/gpt-4o
```

The reasoning model runs the read-only **planner** and **explainer** agents. The coding model runs the **developer** agent, which can edit files and run tests. Each model can use a different provider, endpoint, key, or model identifier. The older `LLM_*` variables are accepted as fallbacks. `--model` overrides only the coding model. Both models must support the tool calls required by their agents.

## CLI help and explanations

Show a quick description of every option, with examples:

```bash
python main.py --help
```

Show the extended guide (no `--repo` or API keys needed):

```bash
python main.py --help-guide
```

The detailed guide explains setup, separate reasoning/coding model variables, the agent workflow, logs, backups, undo, tests, and command examples.

## Run

```bash
python main.py --repo "$HOME/projects/sample-python-project" --message "Explain the code structure; do not edit files"
python main.py --repo "$HOME/projects/sample-python-project" --allow-tests --message "Improve error handling and run pytest"
```

## Configure agents and tasks

- `config/agents.yaml`: roles, goals, backstories, tool permissions, and iteration limits.
- `config/tasks.yaml`: task descriptions, assigned agent names, and expected outputs. Tasks execute in YAML order; previous task context is passed to later tasks by default.
- Task descriptions support `{repo}`, `{request}`, and `{test_instruction}` placeholders.
- Available tool names: `list_python_files`, `read_python_file`, `preview_python_patch`, `patch_python_file`, `run_pytest`.

You can add another agent to `agents.yaml` and assign tasks to it in `tasks.yaml`. Only the developer agent should have `patch_python_file` if you want read-only analysis/review agents. `run_pytest` requires `--allow-tests`.

To use other YAML files:

```bash
python main.py --repo "$HOME/projects/sample-python-project" --agents-config /path/agents.yaml --tasks-config /path/tasks.yaml --message "Fix a bug"
```

## Explanation of completed tasks

After the planner and developer finish, the `explainer` agent runs a separate `explain` task (both configured in YAML). It describes the request, confirmed code changes, rationale, test results, and any unfinished work. Its explanation appears in the terminal and in both Markdown and HTML logs under **EXPLANATION OF WORK COMPLETED**. The report is based on the developer's task output, so review the actual files for critical changes. The explainer has read-only tools and cannot modify code.

You can customize its instructions in `config/tasks.yaml` or its role in `config/agents.yaml`.

## Logs and backups

The standalone agent includes a dedicated `logs/` directory. Each run automatically writes **both** `logs/run-<timestamp>.md` and `logs/run-<timestamp>.html` with the request, configuration summary, console activity, and final result or error. Open the `.html` file in a browser for a readable report; HTML is generated even if the agent fails after logging starts. View the newest logs with:

```bash
ls -lt logs/
less "$(ls -t logs/run-*.md | head -n 1)"
xdg-open "$(ls -t logs/run-*.html | head -n 1)"
```

To use another directory, pass `--log-dir /path/to/logs`.

Backups: **`backups/` inside this standalone agent directory** by default (or `--backup-dir` to override). Both logs and backups remain outside the target sample-python-project project. File tools only edit existing `.py` files and reject symlink escapes and protected directories. Running tests executes project code: use `--allow-tests` only for trusted projects or a sandbox. Do not share `.env` or unreviewed logs containing private source or secrets.

## Undo changes (no Git or API key needed)

Every successful agent patch is recorded in an undo journal inside the standalone `backups/` directory. To undo the most recent run's edits:

```bash
python main.py --repo "$HOME/projects/sample-python-project" --undo
```

See run IDs and undo a specific latest outstanding run:

```bash
python main.py --repo "$HOME/projects/sample-python-project" --list-runs
python main.py --repo "$HOME/projects/sample-python-project" --undo RUN_ID
```

Undo restores all patched Python files from the selected run in reverse edit order. It **refuses** to overwrite a file changed after the agent edited it (checksum conflict), refuses to undo an older run while newer outstanding edits exist, and refuses altered backups. It does not reverse arbitrary side effects from pytest or external commands. Only edits made by this version are journaled; earlier backups cannot be automatically undone through `--undo`. If you used a custom `--backup-dir` for the run, pass the same directory when undoing.

### Execution log formatting (no Markdown package required)

Each run saves `logs/run-<timestamp>.md` and a matching self-contained `.html` file.
The execution transcript is captured as preformatted text so CrewAI's verbose
output, indentation, and code snippets are preserved. Separate sections contain
per-task results, the explainer's report, and the final result. The HTML renderer
uses only the Python standard library and escapes agent/tool output for safety.
No `Markdown` PyPI dependency is needed.

### Undo command displayed after each run

After a coding run, the terminal prints the exact command for undoing **that run**,
including its run ID and repository path. The command is also included in the
Markdown and HTML logs under **Undo this run**. When no edits were recorded,
the output states that there is nothing to undo. A custom `--backup-dir` is
included in the command when supplied.

### Token usage

No maximum token count is imposed by this agent, either per request or per run.
CrewAI provider-reported token usage is displayed when available.



## Python-only scope

The agent lists, reads, previews, and edits only existing `.py` files. Other languages, documentation and configuration files cannot be accessed through the agent file tools. Python patches are AST syntax-checked. `--allowed-paths` and `--read-only-paths` remain available. `--allow-tests` enables pytest on trusted repositories. Backups, undo, HTML/Markdown logs, and separate reasoning/coding models remain supported.

## Model-independent editing

Edits no longer depend on the model being able to call tools. Every model call
still goes through `Crew.kickoff()`, but the host application does the parts
that weaker models get wrong:

- **Context without tool calls.** The host lists the editable Python files (with
  their top-level definitions) for the planner and embeds the current content of
  the file being edited in each developer task.
- **Edits as plain text.** The developer answers with SEARCH/REPLACE blocks. A JSON
  `patches` object or, for a short file, the complete file in one code block are
  also accepted, with prose around them ignored. `edit_protocol.py` locates each
  edit in the current file (tolerating trailing whitespace, line endings and a
  uniform indentation shift, but never an ambiguous match) and applies it through
  the same `patch_project_file` tool, so path rules, syntax checks, backups and
  the undo journal are unchanged.
- **Tools only where they work.** `--tool-mode auto` (default) runs one small
  CrewAI task per model to see whether it really invokes a tool. Models that do
  keep their tools; models that do not get tool-free agents. An attempt that
  fails with tools is repeated without them.
- **Specific retries.** A rejected answer is retried with the exact validation
  error and the current file. Edits already applied are recognised and skipped.

| Option | Behaviour |
| --- | --- |
| `--tool-mode auto` | Probe each model once; use tools only if they work (default) |
| `--tool-mode text` | Never rely on tool calls; works with any chat model, no probe request |
| `--tool-mode tools` | Skip the probe; offer tools first, fall back to text per file |
| `--max-inline-chars N` | Largest file embedded in an edit task (default 60000); lower it for small context windows |
| `--edit-retries N` | Extra tool-free attempts per file after the built-in two |

`python main.py --check-models` now also reports whether each model can call
CrewAI tools. `--legacy-monolithic` keeps the old single-crew run, which still
requires tool calling.

Limits: only existing `.py` files are edited; a file larger than
`--max-inline-chars` can only be edited by a model with working tools; and a
journaled edit proves that a change was written, not that it is correct.

Run the tests (no model or CrewAI installation needed) with `python -m pytest tests`.

---

# Standalone CrewAI Python Code Agent

Edit an existing local Python project using CrewAI. No Git required. The agent saves a backup before each edit and supports safe local undo and plain-text .log files for each run. Agents and tasks are defined in YAML. A read-only reasoning planner prepares a change plan, the coding agent edits files, and a read-only reasoning explainer summarizes the results.

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

The reasoning model runs the read-only **planner** and **explainer** agents. The coding model runs the **developer** agent, which can edit files and run tests. Each model can use a different provider, endpoint, key, or model identifier. The older `LLM_*` variables are accepted as fallbacks. `--model` overrides only the coding model. Neither model needs to support tool calls.

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
- Available tool names: `list_project_files`, `read_project_file`, `preview_project_patch`, `patch_project_file`, `run_pytest`.

You can add another agent to `agents.yaml` and assign tasks to it in `tasks.yaml`. Only the developer agent should have `patch_project_file` if you want read-only analysis/review agents. `run_pytest` requires `--allow-tests`.

To use other YAML files:

```bash
python main.py --repo "$HOME/projects/sample-python-project" --agents-config /path/agents.yaml --tasks-config /path/tasks.yaml --message "Fix a bug"
```

## Explanation of completed tasks

After the planner and developer finish, the `explainer` agent runs a separate `explain` task (both configured in YAML). It describes the request, confirmed code changes, rationale, test results, and any unfinished work. Its explanation appears in the terminal and in both plain-text .log files under EXPLANATION OF WORK COMPLETED. The report is based on the developer's task output, so review the actual files for critical changes. The explainer has read-only tools and cannot modify code.

You can customize its instructions in `config/tasks.yaml` or its role in `config/agents.yaml`.

## Logs and backups

The standalone agent includes a dedicated `logs/` directory. Each run writes one plain-text `logs/run-<timestamp>.log` containing the request, configuration, console transcript, task results, explanation, token usage, and undo command. View the newest logs with:

```bash
ls -lt logs/
less "$(ls -t logs/run-*.log | head -n 1)"
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


The execution transcript is captured as preformatted text so CrewAI's verbose
output, indentation, and code snippets are preserved. Separate sections contain
uses only the Python standard library and escapes agent/tool output for safety.

### Undo command displayed after each run

After a coding run, the terminal prints the exact command for undoing **that run**,
including its run ID and repository path. The command is also included in the
plain-text .log files under UNDO THIS RUN. When no edits were recorded,
the output states that there is nothing to undo. A custom `--backup-dir` is
included in the command when supplied.

### Token usage

No maximum token count is imposed by this agent, either per request or per run.
CrewAI provider-reported token usage is displayed when available.



## Python-only scope


## NVIDIA Nemotron / OpenAI-compatible models

Configure each model separately in `.env` using `REASONING_LLM_*` and
`CODING_LLM_*`. Set `*_LLM_BASE_URL` to the actual NVIDIA-compatible endpoint,
`*_LLM_MODEL` to the model identifier accepted by CrewAI/LiteLLM for that
endpoint, and `*_LLM_API_KEY` to your key. Model identifiers and context sizes
vary by deployment; do not assume an example model's window applies to yours.

Optional settings:

- `REASONING_LLM_CONTEXT_WINDOW` and `CODING_LLM_CONTEXT_WINDOW`: positive
  integer context capacities, passed to CrewAI's `context_window_size`.
- `REASONING_LLM_TEMPERATURE` and `CODING_LLM_TEMPERATURE`: optional sampling
  temperature. A low value (for example `0.1`) helps models copy code exactly.
  Leave unset for models that reject the parameter.
- `REASONING_LLM_EXTRA_BODY` and `CODING_LLM_EXTRA_BODY`: JSON objects forwarded
  as provider-specific `extra_body`. Only use parameters supported by your
  deployed model and endpoint. Do not use this to set token limits.

Test basic connectivity without editing a repository:

```bash
python main.py --check-models
```

This makes one small live inference request per configured model (which may
incur provider charges). It checks authentication, endpoint, model ID, and
basic response compatibility, then runs one tiny CrewAI task per model to see
whether it can call tools. Editing works either way (see "Model-independent
editing"). API/provider failures are reported with the failing model role and
exception.

No custom total-token or output-token budget is enforced.

## Nemotron / tool-call reliability

The developer is instructed to invoke `patch_project_file` with an exact source
fragment and replacement, not just describe a patch. The file tool emits
`FILE_TOOL_EVENT` entries to the console and the timestamped `.log`, recording
attempts, errors and successful writes without logging full source fragments.
After the normal crew completes, the agent checks its backup/undo journal for
actual writes. If there were none, it runs a **developer-only retry** once by
default (`--edit-retries 0` to disable, up to 5). If no edits are recorded
after retry, the command exits with status **2** rather than claiming success.
This applies to coding requests; read-only/explanation requests with no edits
will also exit 2, so set `--edit-retries 0` for those requests.

The retry is another model call and may incur API charges. It cannot overcome
unsupported tool schemas, access restrictions, or API failures. The journal
verifies that writes occurred, not that the requested behavior is correct.
Only `.py` files are editable; path restrictions, backups, undo, and plain
`.log` output are preserved.

## Focused CrewAI execution (default)

The default execution path runs a read-only planning task through `Crew.kickoff()`,
then runs **one focused CrewAI coding task per selected Python file** (up to six by
default). The developer can call `patch_project_file` directly. If a CrewAI task
instead returns a valid JSON `patches` object, Python validates and applies its
exact-text patches through the **same** safe, journaled tool. No standalone LLM
calls or deterministic-model fallback are used by this workflow.

After every coding task the undo journal determines whether a real edit occurred.
The final log records completed/incomplete files and test results. With
`--allow-tests`, pytest runs after the editing tasks. `--max-edit-files 4`
reduces the file limit; `--legacy-monolithic` restores the old single-crew
planner/developer/explainer sequence. `--no-deterministic-fallback` is accepted
for backwards CLI compatibility; the focused workflow never uses that fallback.

The planner must explicitly identify relevant existing `.py` paths; a missing
file list causes the run to stop without inventing targets. Model responses
remain fallible; verify resulting edits and tests before deployment.

### Strict focused-edit validation and retry

The default focused workflow requires the developer to either invoke the patch tool
successfully or return a **single JSON object** with a nonempty `patches` array.
Every patch must contain exactly `filename`, `old_text`, and `new_text`, target the
current selected file, and provide a unique nonempty source fragment. Narrative
success claims, empty arrays, wrong paths, and no-op replacements are rejected.

If an attempt yields no journaled edit, CrewAI runs **one** targeted retry with the
specific validation error. A patch executor response is not considered success
unless a journal entry appears. Partial patch applications are flagged as incomplete
and are not retried automatically; inspect the journal and use the existing undo
command if necessary. The final report separates `verified` (at least one journaled
edit) from `complete` (verified without a remaining patch error). This is structural
validation, **not** proof that the requested feature works; enable `--allow-tests`
for available repository tests and validate `.doc` behavior with a real legacy
binary Word fixture.

The sections "Nemotron / tool-call reliability", "Focused CrewAI execution" and
"Strict focused-edit validation and retry" above describe the earlier
tool-or-strict-JSON behaviour; where they differ, "Model-independent editing" at
the top of this file is current.

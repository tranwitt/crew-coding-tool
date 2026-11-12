# Standalone CrewAI Python Code Agent

Point a CrewAI crew at a local Python project and give it a request. The crew
analyses the code and either edits it or answers a question about it. No Git is
required: every edit is backed up, journaled and undoable, and every run writes a
plain-text log.

## Setup (Linux)

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to configure two independent models:

```dotenv
REASONING_LLM_BASE_URL=https://api.openai.com/v1
REASONING_LLM_API_KEY=your-reasoning-key
REASONING_LLM_MODEL=openai/gpt-4o

CODING_LLM_BASE_URL=https://api.openai.com/v1
CODING_LLM_API_KEY=your-coding-key
CODING_LLM_MODEL=openai/gpt-4o
```

The reasoning model runs the **planner** and **explainer** agents; the coding
model runs the **developer** agent. Each can use a different provider, endpoint,
key or model identifier. The older `LLM_*` variables are accepted as fallbacks,
and `--model` overrides the coding model only. Neither model needs to support
tool calls.

Optional per model (`REASONING_` or `CODING_` prefix):

- `*_LLM_CONTEXT_WINDOW`: positive integer, passed to CrewAI's `context_window_size`.
- `*_LLM_EXTRA_BODY`: JSON object forwarded as provider-specific `extra_body`.
- `*_LLM_TEMPERATURE`: sampling temperature. A low value (for example `0.1`)
  helps models copy code exactly. Leave unset for models that reject it.

No token or output limit is imposed by this tool.

## Run types

There is one pipeline with two run types.

**Edit run** (default): select files → edit each file → run tests → explain.

```bash
python main.py --repo "$HOME/projects/sample" --allow-tests --message "Improve error handling in the loader"
```

**Analysis run** (`--analyze`): select files → answer the request. It never
writes to the repository; the patch tool is withheld from every agent.

```bash
python main.py --repo "$HOME/projects/sample" --analyze --message "Explain how configuration is loaded"
```

| Exit code | Edit run | Analysis run |
| --- | --- | --- |
| 0 | at least one edit was written and journaled | a report was produced |
| 1 | the run crashed | no report was produced, or the run crashed |
| 2 | nothing was changed | (not used) |

The stages:

1. **Select files.** The host lists the Python files with their top-level
   definitions and the planner names the relevant ones. If it names no existing
   file, paths written in the request are used, then a keyword match on file and
   symbol names. Paths are never invented.
2. **Edit** (edit runs). One developer task per file, with the current file
   content and the plan in the task. Up to `--max-edit-files` files (default 6).
3. **Verify** (edit runs). A change counts only when it is in the undo journal.
   With `--allow-tests`, pytest runs after the edits.
4. **Explain / analyse.** The explainer describes an edit run from the
   host-computed diff, per-file outcome and test result, or answers an analysis
   request from the selected files. `--no-explain` skips this stage for edit runs.

## Model-independent editing

Every model call goes through `Crew.kickoff()`, but nothing depends on the model
calling tools:

- **Context without tool calls.** File listings and file contents are embedded in
  the task by the host.
- **Edits as plain text.** The developer answers with SEARCH/REPLACE blocks. A
  JSON `patches` object or, for a short file, the complete file in one code block
  are also accepted; surrounding prose is ignored. Each edit is located in the
  current file (tolerating trailing whitespace, line endings and a uniform
  indentation shift, but never an ambiguous match) and applied through
  `patch_project_file`.
- **Tools only where they work.** `--tool-mode auto` (default) runs one small
  CrewAI task per model to see whether it really invokes a tool. `text` never
  relies on tool calls and skips the probe; `tools` skips the probe and offers
  tools first. A stage that fails with tools is repeated without them.
- **Specific retries.** A rejected answer is retried with the exact validation
  error and the current file; edits already applied are recognised and skipped.
  `--edit-retries N` adds attempts per file.

`--max-inline-chars N` (default 60000) is the largest file embedded in a task;
lower it for small context windows. A larger file can only be edited by a model
with working tools.

Check both models, including tool calling, without touching a repository (this
makes a few small paid requests):

```bash
python main.py --check-models
```

## Configure agents

`config/agents.yaml` defines `planner`, `developer` and `explainer`: role, goal,
backstory, model (`reasoning` or `coding`), tools and `max_iter`. Use another
file with `--agents-config`. `{repo}` and `{request}` are replaced in agent text.

- Tool names: `list_project_files`, `read_project_file`, `preview_project_patch`,
  `patch_project_file`, `run_pytest` (the last needs `--allow-tests`).
- `use_system_prompt: false` is available for models that reject the system role.

The task prompts of each stage live in `pipeline.py`.

## Scope and safety

- Only existing `.py` files are read or edited; the crew cannot create files.
- `--allowed-paths` limits editing to the listed files or directories;
  `--read-only-paths` can be read but never edited.
- Symlink escapes, protected directories, secrets files and binary files are
  rejected. Every patch is syntax-checked before it is written.
- Running tests executes project code: use `--allow-tests` only for trusted
  projects or in a sandbox.
- A journaled edit proves that a change was written, not that it is correct.
  Review the diff and run your tests.

## Logs and backups

Each run writes `logs/run-<timestamp>.log` with the request, configuration,
console transcript, the explanation or analysis report, token usage as reported
by the provider, the verified status and the undo command. Use `--log-dir` for
another directory.

Backups and undo journals go to `backups/` beside this tool (or `--backup-dir`).
Both directories stay outside the target project and may contain private source
code; do not share them or `.env` unreviewed.

## Undo

```bash
python main.py --repo "$HOME/projects/sample" --undo            # latest run
python main.py --repo "$HOME/projects/sample" --list-runs
python main.py --repo "$HOME/projects/sample" --undo RUN_ID
```

Undo restores every file a run patched. It refuses to overwrite a file changed
since the run, to undo an older run while a newer one is outstanding, and to use
an altered backup. It does not reverse side effects of pytest. If the run used
`--backup-dir`, pass the same directory when undoing; the exact command is
printed after each edit run.

## Code layout

| File | Purpose |
| --- | --- |
| `main.py` | Command line, run setup, final report and exit code |
| `pipeline.py` | The stages and the two run types |
| `edit_protocol.py` | Parse text answers into exact, unambiguous edits |
| `repo_context.py` | File listing, file selection, run diffs |
| `crew_setup.py` | Build agents from YAML; decide tool modes |
| `tool_probe.py` | Check whether a model really calls CrewAI tools |
| `models.py` | Model settings from `.env`; `--check-models` |
| `agent_tools.py` | Read, preview, patch and pytest tools |
| `config.py` | Path rules for the target repository |
| `undo.py` | Edit journal and undo |
| `run_log.py` | Console capture and log files |

Run the tests (no model or CrewAI installation needed):

```bash
python -m pytest tests
```

Use `python main.py --help` for all options and `--help-guide` for a longer guide.

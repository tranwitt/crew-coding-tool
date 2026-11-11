"""Model-independent text-to-patch bridge: LLMs never need to call tools.

Models return ordinary JSON text; trusted local code validates and applies edits.
"""
import json
import re
from pathlib import Path


def parse_json(value):
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value, flags=re.I)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for pos, char in enumerate(value):
            if char in '[{':
                try:
                    obj, _ = decoder.raw_decode(value[pos:])
                    return obj
                except json.JSONDecodeError:
                    continue
        raise ValueError('Model response did not contain valid JSON')


def call_text(llm, system, user):
    response = llm.call([{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}])
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get('content') or response)
    return str(getattr(response, 'content', response))


def run_deterministic(cfg, llm, request, patch_tool, *, max_files=8, max_rounds=3):
    """Select files and apply exact replacements through the existing safe patch tool.

    No tool-call support, JSON response mode, or provider-specific API is needed.
    Fail closed on invalid proposals. Existing path/AST/backup/undo checks remain.
    """
    files = []
    for path in sorted(cfg.root.rglob('*.py')):
        if any(part in {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'backups', 'logs', 'build', 'dist'} for part in path.relative_to(cfg.root).parts):
            continue
        try:
            cfg.file(path.relative_to(cfg.root).as_posix(), for_edit=True)
            files.append(path.relative_to(cfg.root).as_posix())
        except (ValueError, OSError, PermissionError):
            continue
        if len(files) >= 1500:
            break
    if not files:
        print('DETERMINISTIC: No writable Python files available.')
        return 0
    system = ('You are a Python code editor. Reply with ONLY valid JSON, without Markdown. '
              'You do not have access to tools. The host application performs all file edits. '
              'Do not claim edits were applied. Choose existing Python files only.')
    selection = parse_json(call_text(llm, system,
        f'User request:\n{request}\n\nAvailable editable Python files:\n' + '\n'.join(files) +
        f'\n\nReturn JSON {{"files": ["path.py", ...]}} with at most {max_files} files most likely needing edits.'))
    selected = selection.get('files', []) if isinstance(selection, dict) else []
    selected = [x for x in selected if isinstance(x, str) and x in files][:max_files]
    if not selected:
        print('DETERMINISTIC: Model selected no valid editable files.')
        return 0
    print('DETERMINISTIC: Selected files:', ', '.join(selected))
    applied = 0
    for filename in selected:
        path = cfg.file(filename, for_edit=True)
        if path.stat().st_size > 100_000:
            print(f'DETERMINISTIC: Skipping oversized file: {filename}')
            continue
        for round_num in range(max_rounds):
            source = path.read_text(encoding='utf-8')
            instruction = (
                f'User request:\n{request}\n\nFilename: {filename}\n'
                'Current complete Python source (the exact text to match):\n'
                f'<source>\n{source}\n</source>\n\n'
                'Return ONLY JSON: {"edits":[{"old_text":"EXACT existing snippet",'
                '"new_text":"replacement snippet"}],"done":true}. '
                'Use short, unique, verbatim old_text snippets. Preserve surrounding code. '
                'Do not escape newlines manually: use valid JSON string escaping. '
                'Return edits:[] when no change is necessary. '
                'Prefer one or several targeted edits, not whole-file replacement. '
                'The host will validate Python syntax and write the changes. '
                + (f'This is round {round_num+1}; previous edits are already in source.' if round_num else '')
            )
            try:
                proposal = parse_json(call_text(llm, system, instruction))
            except Exception as exc:
                print(f'DETERMINISTIC: Invalid model JSON for {filename}: {exc}')
                break
            edits = proposal.get('edits', []) if isinstance(proposal, dict) else []
            if not isinstance(edits, list) or not edits:
                print(f'DETERMINISTIC: No further edits proposed for {filename}')
                break
            successes = 0
            for edit in edits[:12]:
                if not isinstance(edit, dict):
                    continue
                old, new = edit.get('old_text'), edit.get('new_text')
                if not isinstance(old, str) or not isinstance(new, str) or not old or old == new:
                    continue
                # The trusted patch tool enforces all restrictions and journals edits.
                result = patch_tool.run(filename=filename, old_text=old, new_text=new)
                print(f'DETERMINISTIC PATCH {filename}: {str(result)[:700]}')
                if isinstance(result, str) and '"status": "updated"' in result:
                    successes += 1
                    applied += 1
            if not successes or proposal.get('done', True):
                break
    print(f'DETERMINISTIC: Applied {applied} verified patch(es).')
    return applied

"""Model configuration from the environment: one coding and one reasoning model."""
import json
import os
import sys

ROLES = ('coding', 'reasoning')


def _settings(prefix, override=None):
    """Read ``<PREFIX>_LLM_*`` variables, with the legacy ``LLM_*`` fallbacks."""
    def env(name):
        return (os.getenv(f'{prefix}_LLM_{name}') or '').strip()

    context_window = None
    if env('CONTEXT_WINDOW'):
        try:
            context_window = int(env('CONTEXT_WINDOW'))
        except ValueError:
            context_window = 0
        if context_window <= 0:
            raise ValueError(f'{prefix}_LLM_CONTEXT_WINDOW must be a positive integer')
    extra_body = None
    if env('EXTRA_BODY'):
        try:
            extra_body = json.loads(env('EXTRA_BODY'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'{prefix}_LLM_EXTRA_BODY must be valid JSON: {exc}') from exc
        if not isinstance(extra_body, dict):
            raise ValueError(f'{prefix}_LLM_EXTRA_BODY must be a JSON object')
    temperature = None
    if env('TEMPERATURE'):
        try:
            temperature = float(env('TEMPERATURE'))
        except ValueError:
            raise ValueError(f'{prefix}_LLM_TEMPERATURE must be a number') from None
    api_key = env('API_KEY') or (os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or '').strip()
    if not api_key:
        raise ValueError(f'Missing {prefix.lower()} API key: set {prefix}_LLM_API_KEY in .env')
    return {
        'model': override or env('MODEL') or os.getenv('LLM_MODEL', 'openai/gpt-4o'),
        'base_url': env('BASE_URL') or (os.getenv('LLM_BASE_URL') or '').strip() or None,
        'api_key': api_key,
        'context_window': context_window,
        'extra_body': extra_body,
        'temperature': temperature,
    }


def load_model_settings(coding_override=None):
    """Return ``{'coding': settings, 'reasoning': settings}``; raises ValueError."""
    return {'coding': _settings('CODING', coding_override), 'reasoning': _settings('REASONING')}


def create_llm(settings, LLM):
    """Build a CrewAI LLM. No token or output limits are imposed here."""
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


def check_models(llms, probe):
    """Make one small request per model, then probe tool calling. Returns success."""
    ok = True
    for label, llm in llms.items():
        try:
            result = llm.call([{'role': 'user', 'content': 'Reply with exactly: OK'}])
            print(f'{label} model responded: {str(result)[:200]}')
        except Exception as exc:
            ok = False
            print(f'{label} model failed ({type(exc).__name__}): {exc}', file=sys.stderr)
    if not ok:
        return False
    print('Both models responded. Checking CrewAI tool calling (one more small request per model)...')
    for label in llms:
        supported, detail = probe(label)
        print(f'{label} model tool calling: {"works" if supported else "NOT usable"} ({detail}); '
              f'--tool-mode auto will use {"tools" if supported else "the tool-free text protocol"}.')
    print('Runs work in either case; tool calling only decides how the agents reach the repository.')
    return True

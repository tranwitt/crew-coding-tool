"""Model configuration from the environment: one coding and one reasoning model.

OpenAI serves models through two APIs: Chat Completions and the newer Responses
API, and some newer models are only available on the latter. ``*_LLM_API``
selects one; the default ``auto`` checks once per run whether Chat Completions
accepts the model and switches to the Responses API when it does not.
"""
import json
import os
import re
import sys

ROLES = ('coding', 'reasoning')
APIS = ('auto', 'completions', 'responses')

# What OpenAI answers when a model is not served by Chat Completions.
_NEEDS_RESPONSES = re.compile(
    r'v1/responses|responses api|not supported in the v1/chat/completions|not a chat model', re.I)


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
    api = (env('API') or os.getenv('LLM_API') or 'auto').strip().lower()
    if api not in APIS:
        raise ValueError(f'{prefix}_LLM_API must be one of: {", ".join(APIS)}')
    store = None
    if env('STORE'):
        if env('STORE').lower() not in ('true', 'false'):
            raise ValueError(f'{prefix}_LLM_STORE must be true or false')
        store = env('STORE').lower() == 'true'
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
        'api': api,
        'reasoning_effort': env('REASONING_EFFORT') or None,
        'store': store,
    }


def load_model_settings(coding_override=None):
    """Return ``{'coding': settings, 'reasoning': settings}``; raises ValueError."""
    return {'coding': _settings('CODING', coding_override), 'reasoning': _settings('REASONING')}


def is_openai_model(settings):
    """CrewAI routes ``openai/...`` and unprefixed model names to its OpenAI provider."""
    return settings['model'].split('/')[0] == 'openai' or '/' not in settings['model']


def create_llm(settings, LLM, api=None):
    """Build a CrewAI LLM. No token or output limits are imposed here.

    ``api`` overrides ``settings['api']``; only 'responses' changes the request,
    because Chat Completions is CrewAI's default.
    """
    kwargs = {'model': settings['model'], 'api_key': settings['api_key']}
    if (api or settings['api']) == 'responses':
        kwargs['api'] = 'responses'
        if settings['store'] is not None:
            kwargs['store'] = settings['store']
    if settings['reasoning_effort']:
        kwargs['reasoning_effort'] = settings['reasoning_effort']
    if settings['base_url']:
        kwargs['base_url'] = settings['base_url']
    if settings['context_window'] is not None:
        kwargs['context_window_size'] = settings['context_window']
    if settings['extra_body'] is not None:
        kwargs['extra_body'] = settings['extra_body']
    if settings['temperature'] is not None:
        kwargs['temperature'] = settings['temperature']
    return LLM(**kwargs)


def _ping(llm):
    llm.call([{'role': 'user', 'content': 'Reply with exactly: OK'}])


def build_llms(settings, LLM):
    """Create one LLM per role and record the API each uses in ``settings[role]['api_used']``.

    With ``api: auto`` an OpenAI model gets one minimal request on Chat
    Completions. Only an answer saying the model is not served there switches it
    to the Responses API; any other failure is left for the run to report.
    Roles sharing a model and endpoint are checked once.
    """
    llms, resolved = {}, {}
    for role in ROLES:
        config = settings[role]
        key = (config['model'], config['base_url'], config['api'])
        if key not in resolved:
            api, note = config['api'], 'configured'
            if api == 'auto':
                api, note = 'completions', 'default'
                if is_openai_model(config):
                    try:
                        _ping(create_llm(config, LLM, 'completions'))
                        note = 'auto: Chat Completions accepted the model'
                    except Exception as exc:
                        if _NEEDS_RESPONSES.search(str(exc)):
                            api = 'responses'
                            note = f'auto: Chat Completions does not serve this model ({str(exc)[:160]})'
                        else:
                            note = f'auto: check failed, keeping Chat Completions ({type(exc).__name__})'
            resolved[key] = (api, note)
        api, note = resolved[key]
        config['api_used'], config['api_note'] = api, note
        llms[role] = create_llm(config, LLM, api)
    return llms


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

"""Host-side setup: file selection, model settings, tool-mode resolution, agent building."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import Config
from crew_setup import build_agents, resolve_tool_modes
from models import build_llms, load_model_settings
from repo_context import files_matching_request, files_named_in, list_python_files


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        (base / 'repo' / 'src' / 'generated').mkdir(parents=True)
        (base / 'repo' / 'src' / 'good.py').write_text('def parse():\n    pass\n')
        (base / 'repo' / 'src' / 'generated' / 'stubs.py').write_text('X = 1\n')
        self.cfg = Config.create(str(base / 'repo'), str(base / 'backups'),
                                 read_only_paths='src/generated/')

    def test_only_existing_paths_are_selected_once(self):
        plan = 'edit `src/good.py`, `src/missing.py`, ./src/good.py'
        self.assertEqual(files_named_in(plan, self.cfg, 6), ['src/good.py'])

    def test_file_lines_take_precedence_over_an_echoed_listing(self):
        plan = 'Files: src/generated/stubs.py, src/good.py\n- **FILE:** src/good.py - parser'
        self.assertEqual(files_named_in(plan, self.cfg, 6, for_edit=False), ['src/good.py'])

    def test_read_only_files_can_be_read_but_not_selected_for_editing(self):
        self.assertEqual(list_python_files(self.cfg), ['src/good.py'])
        self.assertEqual(list_python_files(self.cfg, for_edit=False),
                         ['src/good.py', 'src/generated/stubs.py'])
        self.assertEqual(files_named_in('src/generated/stubs.py', self.cfg, 6), [])

    def test_keyword_match_uses_paths_and_symbols(self):
        files = ['a/loader.py', 'b/report.py', 'c/x.py', 'd/y.py']
        symbols = {'b/report.py': ['def render_table']}
        self.assertEqual(files_matching_request('Fix the loader', files, symbols, 3), ['a/loader.py'])
        self.assertEqual(files_matching_request('rendering is slow', files, symbols, 3), ['b/report.py'])
        self.assertEqual(files_matching_request('do something', files, symbols, 3), [])


class ModelSettingsTests(unittest.TestCase):
    def test_roles_are_independent_with_legacy_fallbacks(self):
        env = {'LLM_API_KEY': 'shared', 'CODING_LLM_MODEL': 'x/coder', 'CODING_LLM_TEMPERATURE': '0.1',
               'REASONING_LLM_BASE_URL': 'http://r/v1', 'REASONING_LLM_EXTRA_BODY': '{"a": 1}'}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = load_model_settings()
        self.assertEqual(settings['coding']['model'], 'x/coder')
        self.assertEqual(settings['coding']['temperature'], 0.1)
        self.assertEqual(settings['reasoning']['model'], 'openai/gpt-4o')
        self.assertEqual(settings['reasoning']['base_url'], 'http://r/v1')
        self.assertEqual(settings['reasoning']['extra_body'], {'a': 1})
        self.assertEqual(settings['reasoning']['api_key'], 'shared')

    def test_invalid_values_are_named(self):
        for env, message in (({}, 'CODING_LLM_API_KEY'),
                             ({'LLM_API_KEY': 'k', 'CODING_LLM_CONTEXT_WINDOW': '-5'}, 'CONTEXT_WINDOW'),
                             ({'LLM_API_KEY': 'k', 'REASONING_LLM_EXTRA_BODY': '[1]'}, 'JSON object')):
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, message):
                    load_model_settings()


class FakeLLM:
    """Stands in for crewai.LLM: gpt-next-pro is only served by the Responses API."""
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeLLM.created.append(kwargs)

    def call(self, messages):
        if self.kwargs['model'] == 'openai/broken':
            raise RuntimeError('401 invalid api key')
        if 'next-pro' in self.kwargs['model'] and self.kwargs.get('api') != 'responses':
            raise RuntimeError('Error code: 404 - This model is only supported in v1/responses '
                               'and not in v1/chat/completions.')
        return 'OK'


class ApiSelectionTests(unittest.TestCase):
    def build(self, **env):
        FakeLLM.created = []
        with mock.patch.dict(os.environ, {'LLM_API_KEY': 'k', **env}, clear=True):
            settings = load_model_settings()
        return settings, build_llms(settings, FakeLLM)

    def test_auto_switches_only_a_model_that_chat_completions_rejects(self):
        settings, llms = self.build(CODING_LLM_MODEL='openai/gpt-next-pro', REASONING_LLM_MODEL='openai/gpt-5.1')
        self.assertEqual(llms['coding'].kwargs['api'], 'responses')
        self.assertNotIn('api', llms['reasoning'].kwargs)
        self.assertEqual((settings['coding']['api_used'], settings['reasoning']['api_used']),
                         ('responses', 'completions'))
        self.assertIn('v1/responses', settings['coding']['api_note'])

    def test_a_shared_model_is_checked_once(self):
        _, llms = self.build(LLM_MODEL='openai/gpt-next-pro')
        self.assertEqual(len(FakeLLM.created), 3)  # one check, then one LLM per role
        self.assertEqual({llm.kwargs['api'] for llm in llms.values()}, {'responses'})

    def test_explicit_setting_skips_the_check_and_passes_options(self):
        settings, llms = self.build(CODING_LLM_MODEL='openai/gpt-5.1', CODING_LLM_API='responses',
                                    CODING_LLM_STORE='false', CODING_LLM_REASONING_EFFORT='low',
                                    REASONING_LLM_API='completions', REASONING_LLM_STORE='true')
        self.assertEqual(len(FakeLLM.created), 2)
        self.assertEqual(llms['coding'].kwargs, {'model': 'openai/gpt-5.1', 'api_key': 'k', 'api': 'responses',
                                                 'store': False, 'reasoning_effort': 'low'})
        self.assertEqual(llms['reasoning'].kwargs, {'model': 'openai/gpt-4o', 'api_key': 'k'})

    def test_other_failures_and_other_providers_stay_on_chat_completions(self):
        settings, llms = self.build(CODING_LLM_MODEL='openai/broken',
                                    REASONING_LLM_MODEL='nvidia_nim/nemotron')
        self.assertEqual(settings['coding']['api_used'], 'completions')
        self.assertIn('check failed', settings['coding']['api_note'])
        self.assertEqual(len(FakeLLM.created), 3)  # the non-OpenAI model is never pinged
        self.assertNotIn('api', llms['reasoning'].kwargs)

    def test_invalid_api_settings_are_named(self):
        for env, message in (({'CODING_LLM_API': 'chat'}, 'CODING_LLM_API must be one of'),
                             ({'REASONING_LLM_STORE': 'maybe'}, 'REASONING_LLM_STORE')):
            with mock.patch.dict(os.environ, {'LLM_API_KEY': 'k', **env}, clear=True):
                with self.assertRaisesRegex(ValueError, message):
                    load_model_settings()


class CrewSetupTests(unittest.TestCase):
    SPECS = {
        'planner': {'model': 'reasoning', 'role': 'r', 'goal': 'g', 'backstory': 'b', 'tools': ['read']},
        'developer': {'model': 'coding', 'role': 'r', 'goal': 'g', 'backstory': 'b for {repo}',
                      'tools': ['read', 'patch_project_file'], 'use_system_prompt': False},
        'explainer': {'model': 'reasoning', 'role': 'r', 'goal': 'g', 'backstory': 'b'},
    }

    def test_auto_mode_follows_the_probe_per_model(self):
        probed = []

        def probe(role):
            probed.append(role)
            return role == 'coding', 'detail'

        modes = resolve_tool_modes(self.SPECS, list(self.SPECS), 'auto', probe)
        self.assertEqual(modes, {'planner': 'text', 'developer': 'tools', 'explainer': 'text'})
        self.assertEqual(probed, ['reasoning', 'coding'])  # explainer has no tools: no probe
        forced = resolve_tool_modes(self.SPECS, ['developer'], 'text', lambda role: self.fail('no probe expected'))
        self.assertEqual(forced, {'developer': 'text'})

    def test_agents_come_with_and_without_tools(self):
        Agent = lambda **kwargs: kwargs
        lookup = {'read': 'READ', 'patch_project_file': 'PATCH'}
        llms = {'coding': 'C', 'reasoning': 'R'}
        agents, text_agents = build_agents(self.SPECS, llms, lookup, Agent, variables={'{repo}': '/r'})
        self.assertEqual(agents['developer']['tools'], ['READ', 'PATCH'])
        self.assertEqual(agents['developer']['backstory'], 'b for /r')
        self.assertIs(agents['developer']['use_system_prompt'], False)
        self.assertNotIn('use_system_prompt', agents['planner'])
        self.assertEqual(text_agents['developer']['tools'], [])
        self.assertIn('no tools', text_agents['developer']['backstory'])
        read_only, _ = build_agents(self.SPECS, llms, lookup, Agent, withhold=('patch_project_file',))
        self.assertEqual(read_only['developer']['tools'], ['READ'])
        with self.assertRaisesRegex(ValueError, 'unknown tools'):
            build_agents({'planner': dict(self.SPECS['planner'], tools=['nope'])}, llms, lookup, Agent)


if __name__ == '__main__':
    unittest.main()

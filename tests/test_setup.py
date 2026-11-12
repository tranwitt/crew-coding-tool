"""Host-side setup: file selection, model settings, tool-mode resolution, agent building."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import Config
from crew_setup import build_agents, resolve_tool_modes
from models import load_model_settings
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

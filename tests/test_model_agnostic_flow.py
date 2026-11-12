"""End-to-end focused workflow with scripted "models" of differing capability.

Config, the patch tool and the undo journal are real; only CrewAI is replaced,
so these tests run without a model or the crewai package.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import crewai.tools  # noqa: F401
except ImportError:  # Minimal stand-in exposing the tool decorator API used here.
    class _Tool:
        def __init__(self, name, func):
            self.name, self.func, self.description = name, func, func.__doc__

        def run(self, **kwargs):
            return self.func(**kwargs)

    stub, stub_tools = types.ModuleType('crewai'), types.ModuleType('crewai.tools')
    stub_tools.tool = lambda name: (lambda func: _Tool(name, func))
    stub.tools = stub_tools
    sys.modules.update({'crewai': stub, 'crewai.tools': stub_tools})

from agent_tools import make_tools
from config import Config
from focused_crew import run_focused_crew
from undo import ACTIVE_RUN, get_runs, undo_run

APP = (
    'def greet(name):\n'
    '    return "Hello " + name\n'
    '\n'
    'def farewell(name):\n'
    '    return "Bye " + name\n'
)
BLOCK = ('app.py\n<<<<<<< SEARCH\n    return "Hello " + name\n=======\n'
         '    return "Hello, " + name + "!"\n>>>>>>> REPLACE\n')


class FakeAgent:
    def __init__(self, name, tools):
        self.name, self.tools = name, {tool.name: tool for tool in tools}


class FakeTask:
    def __init__(self, description, expected_output, agent):
        self.description, self.agent = description, agent


def crew_for(model, calls):
    """Build a Crew class whose kickoff() asks the scripted model for an answer."""
    class FakeCrew:
        def __init__(self, agents, tasks, process, verbose):
            self.task = tasks[0]

        def kickoff(self):
            calls.append(self.task)
            return types.SimpleNamespace(raw=model(self.task), token_usage=None)
    return FakeCrew


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        (base / 'repo' / 'pkg').mkdir(parents=True)
        (base / 'repo' / 'app.py').write_text(APP)
        (base / 'repo' / 'pkg' / 'util.py').write_text('def unused():\n    return None\n')
        self.app = base / 'repo' / 'app.py'
        self.cfg = Config.create(str(base / 'repo'), str(base / 'backups'))
        self.tools = {tool.name: tool for tool in make_tools(self.cfg)}
        token = ACTIVE_RUN.set('testrun')
        self.addCleanup(ACTIVE_RUN.reset, token)

    def edits(self):
        entry = next((e for e in get_runs(self.cfg) if e['run_id'] == 'testrun'), None)
        return entry['edits'] if entry else []

    def run_flow(self, model, tool_modes, request='Make the greeting friendlier', **extra):
        calls = []
        with_tools = {'planner': FakeAgent('planner', [self.tools['read_project_file']]),
                      'developer': FakeAgent('developer', self.tools.values())}
        without = {name: FakeAgent(name, []) for name in with_tools}
        _, status = run_focused_crew(
            cfg=self.cfg, request=request, agents=with_tools, text_agents=without,
            tool_modes=tool_modes, task_defs={}, render=str, tool_lookup=self.tools,
            max_files=6, allow_tests=False, edits_in_run=self.edits,
            Crew=crew_for(model, calls), Task=FakeTask, Process=types.SimpleNamespace(sequential=1),
            **extra)
        return status, calls

    def test_model_without_tool_calling_edits_through_text(self):
        def model(task):
            self.assertEqual(task.agent.tools, {}, 'text mode must not hand out tools')
            if task.agent.name == 'planner':
                self.assertIn('app.py (5 lines): def greet, def farewell', task.description)
                return 'Thought: easy.\nFILE: app.py - change greet()\n- keep the signature'
            self.assertIn('<current_file>\n' + APP, task.description)
            return 'I updated the greeting as requested.\n\n```\n' + BLOCK + '```\nAll done!'

        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertEqual(len(calls), 2)
        self.assertIn('return "Hello, " + name + "!"', self.app.read_text())
        self.assertEqual(len(self.edits()), 1)
        undo_run(self.cfg, 'testrun')
        self.assertEqual(self.app.read_text(), APP)

    def test_narrative_only_answer_is_retried_with_the_validation_error(self):
        answers = iter(['I have successfully modified app.py.', BLOCK])

        def model(task):
            if task.agent.name == 'planner':
                return 'FILE: app.py - change greet()'
            return next(answers)

        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertIn('PREVIOUS ATTEMPT FAILED VALIDATION', calls[-1].description)
        self.assertEqual(status['reports'][0]['attempts'][0]['result'], 'failed')

    def test_tool_capable_model_patches_directly(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'FILE: app.py - change greet()'
            task.agent.tools['patch_project_file'].run(
                filename='app.py', old_text='"Hello " + name', new_text='"Hi " + name')
            return 'Patched.'

        status, calls = self.run_flow(model, {'planner': 'tools', 'developer': 'tools'})
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(status['tool_modes']['developer_final'], 'tools')

    def test_failing_tool_mode_falls_back_to_text(self):
        def model(task):
            if task.agent.tools:
                raise RuntimeError('400: this endpoint does not support tools')
            if task.agent.name == 'planner':
                return 'FILE: app.py - change greet()'
            return BLOCK

        status, calls = self.run_flow(model, {'planner': 'tools', 'developer': 'tools'})
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertEqual(status['tool_modes']['developer_final'], 'text')
        self.assertEqual([bool(c.agent.tools) for c in calls], [True, False, True, False])

    def test_partial_application_retries_only_the_missing_edit(self):
        bad = ('<<<<<<< SEARCH\n    return "Goodbye " + name\n=======\n'
               '    return "See you, " + name\n>>>>>>> REPLACE\n')
        good = bad.replace('Goodbye', 'Bye')
        answers = iter([BLOCK + bad, BLOCK + good])

        def model(task):
            if task.agent.name == 'planner':
                return 'FILE: app.py'
            return next(answers)

        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['completed_files'], ['app.py'])
        text = self.app.read_text()
        self.assertEqual(text.count('return "Hello, " + name + "!"'), 1)
        self.assertIn('return "See you, " + name', text)
        self.assertEqual(len(self.edits()), 2)
        self.assertIn('already includes 1 edit(s)', calls[-1].description)

    def test_useless_planner_still_reaches_the_right_file(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'You should modify the greeting module at src/greetings.py.'
            self.assertIn('Edit ONLY app.py', task.description)
            return BLOCK

        status, _ = self.run_flow(model, {'planner': 'text', 'developer': 'text'},
                                  request='Make greet friendlier')
        self.assertIn('keyword match', status['selection_source'])
        self.assertEqual(status['completed_files'], ['app.py'])

    def test_no_change_answer_is_not_reported_as_failure(self):
        def model(task):
            return 'FILE: pkg/util.py' if task.agent.name == 'planner' else 'NO_CHANGES_NEEDED'

        status, _ = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['unchanged_files'], ['pkg/util.py'])
        self.assertEqual(status['incomplete_files'], [])
        self.assertEqual(self.edits(), [])

    def test_syntax_breaking_edit_is_rejected_by_the_patch_tool(self):
        broken = '<<<<<<< SEARCH\ndef greet(name):\n=======\ndef greet(name:\n>>>>>>> REPLACE\n'

        def model(task):
            return 'FILE: app.py' if task.agent.name == 'planner' else broken

        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['incomplete_files'], ['app.py'])
        self.assertEqual(self.app.read_text(), APP)
        self.assertIn('SyntaxError', status['reports'][0]['errors'][0])
        self.assertEqual(len(calls), 3)  # plan + two text attempts


if __name__ == '__main__':
    unittest.main()

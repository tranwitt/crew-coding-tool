"""End-to-end pipelines with scripted "models" of differing capability.

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
from pipeline import CrewRunner, run_analysis_pipeline, run_edit_pipeline
from task_config import load_tasks
from undo import ACTIVE_RUN, get_runs, undo_run

TASKS = load_tasks(Path(__file__).resolve().parent.parent / 'config' / 'tasks.yaml',
                   {'planner', 'developer', 'explainer'})

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


class PipelineCase(unittest.TestCase):
    ALLOW_TESTS = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        (base / 'repo' / 'pkg').mkdir(parents=True)
        (base / 'repo' / 'app.py').write_text(APP)
        (base / 'repo' / 'pkg' / 'util.py').write_text('def unused():\n    return None\n')
        self.app = base / 'repo' / 'app.py'
        self.cfg = Config.create(str(base / 'repo'), str(base / 'backups'))
        self.tools = {tool.name: tool for tool in make_tools(self.cfg, allow_tests=self.ALLOW_TESTS)}
        token = ACTIVE_RUN.set('testrun')
        self.addCleanup(ACTIVE_RUN.reset, token)

    def edits(self):
        entry = next((e for e in get_runs(self.cfg) if e['run_id'] == 'testrun'), None)
        return entry['edits'] if entry else []

    def runner(self, model, tool_modes, calls):
        read_only = [self.tools['read_project_file']]
        with_tools = {'planner': FakeAgent('planner', read_only),
                      'developer': FakeAgent('developer', self.tools.values()),
                      'explainer': FakeAgent('explainer', read_only)}
        without = {name: FakeAgent(name, []) for name in with_tools}
        return CrewRunner(agents=with_tools, text_agents=without, tool_modes=tool_modes,
                          Crew=crew_for(model, calls), Task=FakeTask,
                          Process=types.SimpleNamespace(sequential=1))

    def run_flow(self, model, tool_modes, request='Make the greeting friendlier', explain=False, tasks=TASKS, **extra):
        calls = []
        status = run_edit_pipeline(
            cfg=self.cfg, request=request, runner=self.runner(model, tool_modes, calls), tasks=tasks,
            patch_tool=self.tools['patch_project_file'], create_tool=self.tools['create_project_file'],
            pytest_tool=self.tools['run_pytest'], allow_tests=self.ALLOW_TESTS,
            edits_in_run=self.edits, max_files=6, explain=explain, **extra)
        return status, calls


class FlowTests(PipelineCase):
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
        self.assertEqual(status['tool_modes']['developer'], 'tools')

    def test_failing_tool_mode_falls_back_to_text(self):
        def model(task):
            if task.agent.tools:
                raise RuntimeError('400: this endpoint does not support tools')
            if task.agent.name == 'planner':
                return 'FILE: app.py - change greet()'
            return BLOCK

        status, calls = self.run_flow(model, {'planner': 'tools', 'developer': 'tools'})
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertEqual(status['tool_modes']['developer'], 'text')
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


    def test_explanation_is_built_from_the_verified_diff(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'FILE: app.py'
            if task.agent.name == 'developer':
                return BLOCK
            self.assertEqual(task.agent.tools, {})
            seen.append(task.description)
            return '## Task overview\nGreeting changed.'

        seen = []
        status, _ = self.run_flow(model, {'planner': 'text', 'developer': 'text'}, explain=True)
        self.assertEqual(status['report'], '## Task overview\nGreeting changed.')
        self.assertIn('+    return "Hello, " + name + "!"', seen[0])
        self.assertIn('- app.py: changed', seen[0])

    # ---- new files

    NEW_FILE = 'def polite(name):\n    return "Dear " + name\n'
    USE_NEW = ('<<<<<<< SEARCH\ndef greet(name):\n    return "Hello " + name\n=======\n'
               'from pkg.manners import polite\n\n\ndef greet(name):\n    return "Hello " + polite(name)\n'
               '>>>>>>> REPLACE\n')

    def creating_model(self, create_answers):
        answers = iter(create_answers)

        def model(task):
            if task.agent.name == 'planner':
                self.assertIn('NEW: path/to/new_file.py', task.description)
                return ('FILE: app.py - use the helper\nNEW: pkg/manners.py - polite() helper\n'
                        'NEW: ../outside.py - nope\nNEW: app.py - exists\nNEW: notes.txt - not python')
            if 'Write the new file' in task.description:
                return next(answers)
            self.assertIn('<file path="pkg/manners.py">\n' + self.NEW_FILE, task.description)
            return self.USE_NEW
        return model

    def test_new_file_is_created_journaled_and_undone(self):
        model = self.creating_model(['Here it is:\n```python\n' + self.NEW_FILE + '```\nCreated!'])
        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'}, explain=True)
        new = self.cfg.root / 'pkg' / 'manners.py'
        self.assertEqual(status['new_files'], ['pkg/manners.py'])  # invalid NEW lines are ignored
        self.assertEqual(status['created_files'], ['pkg/manners.py'])
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertEqual(new.read_text(), self.NEW_FILE)
        self.assertIn('/dev/null (new file)', calls[-1].description)
        self.assertIn('- pkg/manners.py: created', calls[-1].description)
        result = undo_run(self.cfg, 'testrun')
        self.assertFalse(new.exists())
        self.assertEqual(result['removed_files'], ['pkg/manners.py'])
        self.assertEqual(self.app.read_text(), APP)

    def test_new_directories_are_removed_again_on_undo(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'NEW: deep/er/mod.py - new module'
            return '```python\nX = 1\n```'

        status, _ = self.run_flow(model, {'planner': 'text', 'developer': 'text'})
        self.assertEqual(status['created_files'], ['deep/er/mod.py'])
        self.assertEqual(status['selected_files'], [])
        self.assertNotIn('error', status)
        undo_run(self.cfg, 'testrun')
        self.assertFalse((self.cfg.root / 'deep').exists())

    def test_unusable_new_file_answer_is_retried(self):
        model = self.creating_model(['I created pkg/manners.py with a polite() helper.',
                                     '```python\ndef polite(name:\n```',
                                     '```python\n' + self.NEW_FILE + '```'])
        status, calls = self.run_flow(model, {'planner': 'text', 'developer': 'text'}, extra_attempts=1)
        self.assertEqual(status['created_files'], ['pkg/manners.py'])
        attempts = status['reports'][0]['attempts']
        self.assertEqual([a['result'] for a in attempts], ['failed', 'failed', 'journal_verified'])
        self.assertIn('exactly one', attempts[0]['error'])
        self.assertIn('not valid Python', attempts[1]['error'])

    def test_tool_capable_model_creates_through_the_tool(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'NEW: pkg/manners.py - helper'
            task.agent.tools['create_project_file'].run(filename='pkg/manners.py', content=self.NEW_FILE)
            return 'Created.'

        status, _ = self.run_flow(model, {'planner': 'tools', 'developer': 'tools'})
        self.assertEqual(status['created_files'], ['pkg/manners.py'])
        self.assertEqual(status['tool_modes']['developer'], 'tools')

    def test_create_tool_refuses_existing_and_protected_paths(self):
        create = self.tools['create_project_file']
        self.assertIn('already exists', create.run(filename='app.py', content='X = 1\n'))
        for name in ('../evil.py', '.git/hook.py', 'notes.txt', '.env'):
            self.assertTrue(create.run(filename=name, content='X = 1\n').startswith('ERROR'), name)
        self.assertIn('SyntaxError', create.run(filename='bad.py', content='def (:\n'))
        self.assertFalse((self.cfg.root / 'bad.py').exists())
        self.assertEqual(self.edits(), [])

    def test_removing_the_create_stage_forbids_new_files(self):
        tasks = {name: stage for name, stage in TASKS.items() if name != 'create'}

        def model(task):
            if task.agent.name == 'planner':
                self.assertNotIn('NEW:', task.description)
                return 'FILE: app.py\nNEW: pkg/manners.py - helper'
            return BLOCK

        status, _ = self.run_flow(model, {'planner': 'text', 'developer': 'text'}, tasks=tasks)
        self.assertEqual(status['new_files'], [])
        self.assertFalse((self.cfg.root / 'pkg' / 'manners.py').exists())

    def test_analysis_run_reports_and_never_writes(self):
        def model(task):
            if task.agent.name == 'planner':
                self.assertIn('must be read', task.description)
                return 'FILE: app.py - holds the greeting'
            self.assertEqual(task.agent.name, 'explainer')
            self.assertIn('<file path="app.py">\n' + APP, task.description)
            return 'greet() concatenates "Hello " and the name.'

        calls = []
        status = run_analysis_pipeline(
            cfg=self.cfg, request='How does greet work?', max_files=6, tasks=TASKS,
            runner=self.runner(model, {'planner': 'text', 'explainer': 'text'}, calls))
        self.assertEqual(status['report'], 'greet() concatenates "Hello " and the name.')
        self.assertIsNone(status['error'])
        self.assertEqual(self.app.read_text(), APP)
        self.assertEqual(self.edits(), [])

    def test_custom_tasks_change_agent_wording_and_stages(self):
        import yaml
        data = yaml.safe_load((Path(__file__).resolve().parent.parent / 'config' / 'tasks.yaml').read_text())
        data['tasks']['select']['agent'] = 'explainer'
        data['tasks']['edit']['description'] = 'HOUSE STYLE: tabs are forbidden.\n' + data['tasks']['edit']['description']
        del data['tasks']['explain']
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'tasks.yaml').write_text(yaml.safe_dump(data))
            tasks = load_tasks(Path(tmp) / 'tasks.yaml', {'planner', 'developer', 'explainer'})

        def model(task):
            if task.agent.name == 'explainer':
                return 'FILE: app.py'
            self.assertTrue(task.description.startswith('HOUSE STYLE: tabs are forbidden.'))
            return BLOCK

        status, calls = self.run_flow(model, {'explainer': 'text', 'developer': 'text'},
                                      explain=True, tasks=tasks)
        self.assertEqual([c.agent.name for c in calls], ['explainer', 'developer'])
        self.assertEqual(status['completed_files'], ['app.py'])
        self.assertIsNone(status['report'])  # no explain stage configured

    def test_analysis_failure_is_reported_without_a_report(self):
        def model(task):
            if task.agent.name == 'planner':
                return 'FILE: app.py'
            raise RuntimeError('503 upstream unavailable')

        status = run_analysis_pipeline(
            cfg=self.cfg, request='How does greet work?', max_files=6, tasks=TASKS,
            runner=self.runner(model, {'planner': 'text', 'explainer': 'tools'}, []))
        self.assertIsNone(status['report'])
        self.assertIn('503', status['error'])


class RepairTests(PipelineCase):
    """Runs real pytest in the temporary repository."""
    ALLOW_TESTS = True
    BREAK = ('<<<<<<< SEARCH\n    return "Hello " + name\n=======\n'
             '    return "Hello, " + nam\n>>>>>>> REPLACE\n')
    FIX = BREAK.replace('"Hello " + name', '"Hello, " + nam').replace('"Hello, " + nam\n>', '"Hello, " + name\n>')

    def setUp(self):
        super().setUp()
        (self.cfg.root / 'tests').mkdir()
        (self.cfg.root / 'tests' / 'test_app.py').write_text(
            'import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent.parent))\n'
            'from app import greet\n\ndef test_greet():\n    assert greet("Ann").startswith("Hello")\n')

    def scripted(self, developer_answers):
        answers = iter(developer_answers)
        return lambda task: 'FILE: app.py' if task.agent.name == 'planner' else next(answers)

    def test_failing_tests_are_fed_back_and_fixed(self):
        status, calls = self.run_flow(self.scripted([self.BREAK, self.FIX]),
                                      {'planner': 'text', 'developer': 'text'})
        self.assertIs(status['tests_passed_before'], True)
        self.assertIs(status['tests_passed'], True)
        self.assertEqual(len(status['repairs']), 1)
        self.assertEqual(status['repairs'][0]['tests_exit_code'], 0)
        self.assertIn('TESTS FAILED after the changes of this run', calls[-1].description)
        self.assertIn("NameError: name 'nam' is not defined", calls[-1].description)
        self.assertIn('return "Hello, " + name\n', self.app.read_text())
        undo_run(self.cfg, 'testrun')
        self.assertEqual(self.app.read_text(), APP)

    def test_unrepaired_failure_is_reported_as_failing(self):
        status, calls = self.run_flow(self.scripted([self.BREAK, 'NO_CHANGES_NEEDED']),
                                      {'planner': 'text', 'developer': 'text'}, repair_rounds=2)
        self.assertIs(status['tests_passed'], False)
        self.assertEqual(len(status['repairs']), 1)  # nothing changed, so no second round
        self.assertEqual(len(calls), 3)

    def test_passing_tests_need_no_repair_and_repair_can_be_disabled(self):
        status, calls = self.run_flow(self.scripted([BLOCK]), {'planner': 'text', 'developer': 'text'})
        self.assertIs(status['tests_passed'], True)
        self.assertEqual(status['repairs'], [])
        undo_run(self.cfg, 'testrun')
        status, calls = self.run_flow(self.scripted([self.BREAK]),
                                      {'planner': 'text', 'developer': 'text'}, repair_rounds=0)
        self.assertIs(status['tests_passed'], False)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()

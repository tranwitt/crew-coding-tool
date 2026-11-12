"""tasks.yaml decides which agent runs each stage and what it is told."""
import tempfile
import types
import unittest
from pathlib import Path

from task_config import fill, load_tasks

DEFAULT = Path(__file__).resolve().parent.parent / 'config' / 'tasks.yaml'
AGENTS = {'planner', 'developer', 'explainer'}


def load_text(text, agents=AGENTS):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'tasks.yaml'
        path.write_text(text)
        return load_tasks(path, agents)


def with_stage(name, **changes):
    """The default configuration with one stage's fields replaced."""
    import yaml
    data = yaml.safe_load(DEFAULT.read_text())
    data['tasks'][name].update(changes)
    return yaml.safe_dump(data)


class TaskConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid_and_complete(self):
        self.assertEqual(set(load_tasks(DEFAULT, AGENTS)), {'select', 'create', 'edit', 'explain', 'analyze'})

    def test_inserted_text_is_never_rescanned(self):
        source = 'print(f"{request} and {filename}")'
        self.assertEqual(fill('{request}: {file_content}', {'request': 'R', 'file_content': source}),
                         'R: ' + source)

    def test_tool_guidance_follows_the_mode(self):
        stage = load_tasks(DEFAULT, AGENTS)['edit']
        values = dict(repo='/r', request='q', filename='a.py', file_content='C', edit_format='F',
                      feedback='', plan='P', created_files='(none)')
        self.assertIn('patch_project_file(filename', stage.render('tools', **values))
        self.assertIn('You have NO tools', stage.render('text', **values))
        self.assertNotIn('{', stage.render('text', **values))

    def test_mistakes_are_reported_with_the_stage_name(self):
        cases = [
            (with_stage('edit', agent='reviewer'), "Stage edit: agent 'reviewer' is not defined"),
            (with_stage('edit', description='Edit {filename} {file_content} {feedback}'), r"must contain \['\{edit_format\}'\]"),
            (with_stage('select', description='{listing} {filname}'), r"unknown placeholders \['\{filname\}'\]"),
            (with_stage('explain', expected_output=''), 'Stage explain: expected_output'),
            (with_stage('analyze', tool_guidance={'always': 'x'}), 'Stage analyze: tool_guidance'),
            ('tasks:\n  review:\n    agent: planner\n', "unknown stages \\['review'\\]"),
            ('agents: {}\n', "nonempty mapping under 'tasks'"),
        ]
        for text, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                load_text(text)


if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deterministic_edits import run_deterministic


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def call(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({'files': ['example.py']})
        return json.dumps({'edits': [{'old_text': 'VALUE = 1', 'new_text': 'VALUE = 2'}], 'done': True})


class FakeConfig:
    def __init__(self, root):
        self.root = root

    def file(self, filename, for_edit=False):
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root) or path.suffix != '.py' or not path.is_file():
            raise ValueError('not allowed')
        return path


class FakePatch:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, filename, old_text, new_text):
        path = self.cfg.file(filename, for_edit=True)
        original = path.read_text()
        if original.count(old_text) != 1:
            return 'ERROR: no unique match'
        path.write_text(original.replace(old_text, new_text))
        return json.dumps({'status': 'updated'})


class DirectEditorTest(unittest.TestCase):
    def test_edits_in_two_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'example.py').write_text('VALUE = 1\n')
            cfg = FakeConfig(root)
            llm = FakeLLM()
            self.assertEqual(run_deterministic(cfg, llm, 'Change value', FakePatch(cfg)), 1)
            self.assertEqual(llm.calls, 2)
            self.assertEqual((root / 'example.py').read_text(), 'VALUE = 2\n')


if __name__ == '__main__':
    unittest.main()

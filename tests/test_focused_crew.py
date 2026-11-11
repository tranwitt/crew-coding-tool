import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from focused_crew import _json_patches, _selected_files


class FocusedTests(unittest.TestCase):
    def test_json_patches_reject_wrong_file(self):
        self.assertEqual(_json_patches('{"patches":[{"filename":"other.py","old_text":"a","new_text":"b"}]}', 'good.py'), [])

    def test_json_patches_accept_fenced(self):
        raw = '```json\n{"patches":[{"filename":"good.py","old_text":"a","new_text":"b"}]}\n```'
        self.assertEqual(len(_json_patches(raw, 'good.py')), 1)

    def test_file_selection_checks_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'good.py').write_text('pass')
            class Config:
                def file(self, name, for_edit=False):
                    p = root / name
                    if not p.is_file():
                        raise FileNotFoundError(name)
                    return p
            self.assertEqual(_selected_files('edit `good.py`, `missing.py`, `good.py`', Config(), 6), ['good.py'])


if __name__ == '__main__':
    unittest.main()

class StrictPatchTests(unittest.TestCase):
    def test_reject_prose_and_empty(self):
        from focused_crew import _parse_patches
        self.assertIsNotNone(_parse_patches('Applied changes successfully', 'x.py')[1])
        self.assertIsNotNone(_parse_patches('{"patches":[]}', 'x.py')[1])

    def test_reject_mixed_targets_and_noop(self):
        from focused_crew import _parse_patches
        self.assertIsNotNone(_parse_patches('{"patches":[{"filename":"x.py","old_text":"a","new_text":"b"},{"filename":"y.py","old_text":"c","new_text":"d"}]}', 'x.py')[1])
        self.assertIsNotNone(_parse_patches('{"patches":[{"filename":"x.py","old_text":"a","new_text":"a"}]}', 'x.py')[1])

    def test_patch_requires_journal_confirmation(self):
        from focused_crew import _apply_patches
        class Tool:
            def run(self, **kwargs):
                return 'ERROR: zero matches'
        applied, error = _apply_patches([{'filename':'x.py','old_text':'a','new_text':'b'}], 'x.py', Tool(), lambda: [])
        self.assertEqual(applied, 0)
        self.assertIn('not journaled', error)

"""Files written on Windows: CRLF line endings, and paths with backslashes."""
import tempfile
import unittest
from pathlib import Path

from agent_tools import make_tools
from config import Config
from focused_crew import _selected_files
from undo import ACTIVE_RUN, undo_run

OLD = '    total = sum(values)\n    return total / len(values)'          # what read_project_file shows
NEW = '    return sum(values) / len(values) if values else 0'


class WindowsFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.root, backups = tmp / 'project', tmp / 'backups'
        (self.root / 'tests').mkdir(parents=True)
        (self.root / 'calc.py').write_bytes(b'def average(values):\r\n    total = sum(values)\r\n    return total / len(values)\r\n')
        (self.root / 'tests' / 'test_calc.py').write_bytes(b'x = 1\r\n')
        self.cfg = Config.create(str(self.root), str(backups))
        self.tools = {t.name: t for t in make_tools(self.cfg)}
        self._run = ACTIVE_RUN.set('test-run')

    def tearDown(self):
        ACTIVE_RUN.reset(self._run)
        self._tmp.cleanup()

    def test_read_shows_lf_so_a_multi_line_edit_applies_to_a_crlf_file(self):
        self.assertNotIn('\r', self.tools['read_project_file'].run(filename='calc.py'))
        result = self.tools['patch_project_file'].run(filename='calc.py', old_text=OLD, new_text=NEW)
        self.assertNotIn('found 0', result)
        self.assertEqual((self.root / 'calc.py').read_bytes(),
                         b'def average(values):\r\n    return sum(values) / len(values) if values else 0\r\n')

    def test_an_lf_file_stays_lf(self):
        (self.root / 'lf.py').write_bytes(b'A = 1\nB = 2\n')
        self.tools['patch_project_file'].run(filename='lf.py', old_text='A = 1\nB = 2', new_text='A = 3\nB = 4')
        self.assertEqual((self.root / 'lf.py').read_bytes(), b'A = 3\nB = 4\n')

    def test_undo_restores_the_crlf_bytes(self):
        before = (self.root / 'calc.py').read_bytes()
        self.tools['patch_project_file'].run(filename='calc.py', old_text=OLD, new_text=NEW)
        undo_run(self.cfg)
        self.assertEqual((self.root / 'calc.py').read_bytes(), before)

    def test_listed_paths_use_forward_slashes(self):
        listed = self.tools['list_project_files'].run()
        self.assertIn('tests/test_calc.py', listed)
        self.assertNotIn('tests\\test_calc.py', listed)

    def test_a_plan_with_backslash_paths_keeps_files_in_subfolders(self):
        self.assertEqual(_selected_files('1. **tests\\test_calc.py**: fix it', self.cfg, 5), ['tests/test_calc.py'])

    def test_a_python_file_with_a_nul_byte_is_refused(self):
        (self.root / 'bad.py').write_bytes(b'A = 1\x00\n')
        with self.assertRaises(ValueError):
            self.cfg.file('bad.py')
        self.cfg.file('calc.py')                                        # a normal file still passes


if __name__ == '__main__':
    unittest.main()

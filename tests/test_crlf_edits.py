"""Edits to files saved with Windows line endings (CRLF).

read_project_file shows the model \n, the edit stage shows the raw text with
\r\n, and models often drop the \r in their JSON. An old_text in either form
must match, and the file must keep its line endings.
"""
import pytest

from agent_tools import make_tools
from config import Config
from undo import ACTIVE_RUN, undo_run

CRLF = b'def average(values):\r\n    total = sum(values)\r\n    return total / len(values)\r\n'
FIXED = b'def average(values):\r\n    return sum(values) / len(values) if values else 0\r\n'
NEW = '    return sum(values) / len(values) if values else 0'


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'calc.py').write_bytes(CRLF)
    cfg = Config.create(str(root), str(tmp_path / 'backups'))
    token = ACTIVE_RUN.set('test-run')
    yield root, cfg, {t.name: t for t in make_tools(cfg)}
    ACTIVE_RUN.reset(token)


@pytest.mark.parametrize('newline', ['\n', '\r\n'], ids=['model-dropped-cr', 'model-kept-cr'])
def test_a_multi_line_edit_to_a_crlf_file_applies_and_keeps_crlf(project, newline):
    root, _, tools = project
    old = f'    total = sum(values){newline}    return total / len(values)'
    result = tools['patch_project_file'].run(filename='calc.py', old_text=old, new_text=NEW)
    assert '"status": "updated"' in result, result
    assert (root / 'calc.py').read_bytes() == FIXED


def test_preview_and_patch_agree_on_a_crlf_file(project):
    _, _, tools = project
    old = '    total = sum(values)\n    return total / len(values)'
    assert not tools['preview_project_patch'].run(filename='calc.py', old_text=old, new_text=NEW).startswith('ERROR')
    assert '"status": "updated"' in tools['patch_project_file'].run(filename='calc.py', old_text=old, new_text=NEW)


def test_undo_restores_the_crlf_bytes(project):
    root, cfg, tools = project
    tools['patch_project_file'].run(filename='calc.py', old_text='    total = sum(values)\n    return total / len(values)', new_text=NEW)
    undo_run(cfg)
    assert (root / 'calc.py').read_bytes() == CRLF


def test_lf_and_mixed_files_are_left_as_they_are(project):
    root, _, tools = project
    (root / 'lf.py').write_bytes(b'A = 1\nB = 2\n')
    tools['patch_project_file'].run(filename='lf.py', old_text='A = 1\nB = 2', new_text='A = 3\nB = 4')
    assert (root / 'lf.py').read_bytes() == b'A = 3\nB = 4\n'
    (root / 'mixed.py').write_bytes(b'A = 1\nB = 2\r\n')       # what a PowerShell here-string writes
    tools['patch_project_file'].run(filename='mixed.py', old_text='A = 1\nB = 2', new_text='A = 3\nB = 4')
    assert (root / 'mixed.py').read_bytes() == b'A = 3\nB = 4\r\n'

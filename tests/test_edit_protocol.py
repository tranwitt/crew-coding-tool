import unittest

from edit_protocol import EditError, NO_CHANGE_SENTINEL, parse_edits, resolve_edit

SOURCE = (
    'import os\n'
    '\n'
    'class Loader:\n'
    '    def load(self, path):\n'
    '        if path.endswith(".txt"):\n'
    '            return open(path).read()\n'
    '        raise ValueError(path)\n'
    '\n'
    'def helper():\n'
    '    return 1\n'
)


def apply(source, raw, filename='app.py'):
    edits, no_change = parse_edits(raw, filename, source)
    for edit in edits:
        resolved = resolve_edit(source, edit)
        if resolved is None:
            continue
        old, new = resolved
        assert source.count(old) == 1, 'patch tool requires a unique exact match'
        source = source.replace(old, new, 1)
    return source, no_change


class SearchReplaceTests(unittest.TestCase):
    def test_block_surrounded_by_prose_and_fences(self):
        raw = (
            'Thought: I will update the helper.\n\nHere is the change:\n\n'
            '```python\napp.py\n<<<<<<< SEARCH\ndef helper():\n    return 1\n=======\n'
            'def helper():\n    return 2\n>>>>>>> REPLACE\n```\nThe file was updated.'
        )
        result, _ = apply(SOURCE, raw)
        self.assertIn('    return 2\n', result)
        self.assertNotIn('    return 1\n', result)

    def test_indentation_shift_is_corrected(self):
        raw = (
            '<<<<<<< SEARCH\nif path.endswith(".txt"):\n    return open(path).read()\n=======\n'
            'if path.endswith((".txt", ".md")):\n    return open(path).read()\n>>>>>>> REPLACE'
        )
        result, _ = apply(SOURCE, raw)
        self.assertIn('        if path.endswith((".txt", ".md")):\n            return open(path).read()\n', result)
        compile(result, 'app.py', 'exec')

    def test_trailing_whitespace_and_crlf_are_tolerated(self):
        source = SOURCE.replace('\n', '\r\n')
        raw = '<<<<<<< SEARCH\ndef helper():   \n    return 1\n=======\ndef helper():\n    return 3\n>>>>>>> REPLACE'
        result, _ = apply(source, raw)
        self.assertIn('    return 3\r\n', result)
        self.assertNotIn('3\n\n', result)
        self.assertEqual(result.count('\r\n'), result.count('\n'))

    def test_ambiguous_and_missing_search_are_rejected(self):
        source = 'x = 1\nprint(x)\nx = 1\n'
        edits, _ = parse_edits('<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE', 'app.py', source)
        with self.assertRaisesRegex(EditError, 'matches 2 places'):
            resolve_edit(source, edits[0])
        edits, _ = parse_edits('<<<<<<< SEARCH\ny = 5\n=======\ny = 6\n>>>>>>> REPLACE', 'app.py', source)
        with self.assertRaisesRegex(EditError, 'not found'):
            resolve_edit(source, edits[0])

    def test_non_line_aligned_duplicate_gets_context(self):
        source = 'xx = 1\nprint("a")\nx = 1\n'
        result, _ = apply(source, '<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE')
        self.assertEqual(result, 'xx = 1\nprint("a")\nx = 2\n')

    def test_wrong_file_unterminated_and_elided_blocks_are_rejected(self):
        with self.assertRaisesRegex(EditError, 'other.py'):
            parse_edits('other.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE', 'app.py', SOURCE)
        with self.assertRaisesRegex(EditError, 'REPLACE'):
            parse_edits('<<<<<<< SEARCH\na\n=======\nb\n', 'app.py', SOURCE)
        with self.assertRaisesRegex(EditError, 'placeholder'):
            parse_edits('<<<<<<< SEARCH\ndef helper():\n    return 1\n=======\n'
                        'def helper():\n    # ... rest unchanged\n>>>>>>> REPLACE', 'app.py', SOURCE)

    def test_repeated_insertion_is_recognised_as_already_applied(self):
        raw = '<<<<<<< SEARCH\nimport os\n=======\nimport os\nimport sys\n>>>>>>> REPLACE'
        once, _ = apply(SOURCE, raw)
        twice, _ = apply(once, raw)
        self.assertEqual(once.count('import sys'), 1)
        self.assertEqual(once, twice)

    def test_repeated_replacement_is_recognised_as_already_applied(self):
        raw = '<<<<<<< SEARCH\n    return 1\n=======\n    return 7\n>>>>>>> REPLACE'
        once, _ = apply(SOURCE, raw)
        edits, _ = parse_edits(raw, 'app.py', once)
        self.assertIsNone(resolve_edit(once, edits[0]))


class OtherFormatTests(unittest.TestCase):
    def test_json_embedded_in_prose(self):
        raw = ('Sure! Here is the patch:\n{"patches":[{"filename":"app.py",'
               '"old_text":"return 1","new_text":"return 5"}]}\nDone.')
        result, _ = apply(SOURCE, raw)
        self.assertIn('    return 5\n', result)

    def test_json_wrong_file_and_empty_are_rejected(self):
        with self.assertRaises(EditError):
            parse_edits('{"patches":[{"filename":"b.py","old_text":"a","new_text":"b"}]}', 'app.py', SOURCE)
        with self.assertRaises(EditError):
            parse_edits('{"patches":[]}', 'app.py', SOURCE)

    def test_whole_file_block(self):
        updated = SOURCE.replace('return 1', 'return 9')
        result, _ = apply(SOURCE, f'Updated file:\n```python\n{updated}```\n')
        self.assertEqual(result, updated)

    def test_fragment_is_not_mistaken_for_whole_file(self):
        with self.assertRaisesRegex(EditError, 'fragment'):
            parse_edits('```python\ndef helper():\n    return 2\n```', 'app.py', SOURCE)

    def test_no_change_sentinel_and_prose(self):
        self.assertEqual(parse_edits(NO_CHANGE_SENTINEL, 'app.py', SOURCE), ([], True))
        with self.assertRaisesRegex(EditError, 'no edits'):
            parse_edits('I have successfully updated app.py.', 'app.py', SOURCE)


if __name__ == '__main__':
    unittest.main()

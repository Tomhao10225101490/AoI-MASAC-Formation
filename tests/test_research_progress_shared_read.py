"""Real temporary-file contracts; no study output, model or environment use."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import serve_research_progress as progress


class SharedReadContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="masac_status_share_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @unittest.skipUnless(os.name == "nt", "Windows ordinary-reader sharing control")
    def test_ordinary_reader_blocks_replace_until_closed(self):
        target, replacement = self.root / "status.json", self.root / "new.json"
        target.write_bytes(b"old")
        replacement.write_bytes(b"new")
        with target.open("rb"):
            with self.assertRaises(PermissionError) as caught:
                os.replace(replacement, target)
            print(f"CONTROL ordinary read blocks atomic replace winerror={caught.exception.winerror}", flush=True)
        os.replace(replacement, target)
        self.assertEqual(target.read_bytes(), b"new")

    def test_shared_reader_replacement_limit_and_close_contract(self):
        target, replacement = self.root / "status.json", self.root / "new.json"
        target.write_bytes(b"old snapshot")
        replacement.write_bytes(b"new snapshot")
        with progress.open_shared_binary(target) as handle:
            try:
                os.replace(replacement, target)
                replaced = True
            except PermissionError as error:
                if os.name != "nt":
                    raise
                self.assertIn(error.winerror, (5, 32))
                replaced = False
                print(f"LIMIT shared destination still blocks replace winerror={error.winerror}", flush=True)
            self.assertEqual(handle.read(), b"old snapshot")
        self.assertTrue(handle.closed)
        if not replaced:
            os.replace(replacement, target)
        self.assertEqual(target.read_bytes(), b"new snapshot")
        print(f"SHARED replacement_while_open={replaced}; replacement_after_close=PASS", flush=True)

    def test_unicode_path_binary_data_and_concurrent_append(self):
        target = self.root / "状态.jsonl"
        original = b'\xef\xbb\xbf{"value":1}\r\n\x00\xff'
        target.write_bytes(original)
        with progress.open_shared_binary(target) as handle:
            with target.open("ab") as writer:
                writer.write(b"appended\r\n")
            self.assertEqual(handle.read(), original + b"appended\r\n")

    def test_missing_path_is_not_created_and_exception_closes_stream(self):
        absent = self.root / "absent.json"
        with self.assertRaises(FileNotFoundError):
            with progress.open_shared_binary(absent):
                self.fail("Missing path must not open")
        self.assertFalse(absent.exists())
        target = self.root / "existing.json"
        target.write_bytes(b"{}")
        with self.assertRaisesRegex(RuntimeError, "consumer error"):
            with progress.open_shared_binary(target) as stream:
                raise RuntimeError("consumer error")
        self.assertTrue(stream.closed)

    def test_document_closes_handle_before_decode_and_next_snapshot(self):
        target, replacement = self.root / "status.json", self.root / "new.json"
        target.write_bytes(b'{"version":0}')
        replacement.write_bytes(b'{"version":1000,"longer":true}')
        original_open = progress.open_shared_binary

        @contextmanager
        def replace_while_open(path):
            with original_open(path) as handle:
                yield handle
            # Windows MoveFileEx replacement may require the target closed,
            # even when FILE_SHARE_DELETE is present. Decode must occur later.
            os.replace(replacement, target)

        reader = progress.Reader()
        with patch.object(progress, "open_shared_binary", replace_while_open):
            first = reader.document(target)
        self.assertEqual(first["data"], {"version": 0})
        self.assertIsNone(first["meta"]["error"])
        second = reader.document(target)
        self.assertEqual(second["data"], {"version": 1000, "longer": True})
        self.assertFalse(second["from_last_good"])

    def test_metrics_partial_line_and_tail_contracts_remain(self):
        target = self.root / "metrics.jsonl"
        target.write_bytes(b'{"kind":"progress","step":1}\n')
        reader = progress.Reader()
        self.assertEqual([r["step"] for r in reader.metrics(target)["rows"]], [1])
        with target.open("ab") as writer:
            writer.write(b'{"kind":"progress","step":2')
        partial = reader.metrics(target)
        self.assertTrue(partial["incomplete_line"])
        self.assertEqual([r["step"] for r in partial["rows"]], [1])
        with target.open("ab") as writer:
            writer.write(b'}\n')
        complete = reader.metrics(target)
        self.assertFalse(complete["incomplete_line"])
        self.assertEqual([r["step"] for r in complete["rows"]], [1, 2])
        tail = reader.tail(target, count=1, parse=True)
        self.assertEqual(tail["rows"], [dict(kind="progress", step=2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)

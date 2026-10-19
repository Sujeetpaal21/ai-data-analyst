import json
import unittest

import pandas as pd

from tests.helpers import excel_bytes
from analyst.ingestion import IngestionError, load_file


class IngestionTests(unittest.TestCase):
    def test_csv_basic_and_semicolon(self):
        lf = load_file("a.csv", b"x,y\n1,2\n3,4\n")
        self.assertEqual(lf.datasets["data"].shape, (2, 2))
        lf = load_file("b.csv", b"x;y\n1;2\n3;4\n")
        self.assertEqual(list(lf.datasets["data"].columns), ["x", "y"])
        self.assertEqual(lf.meta["delimiter"], ";")

    def test_tsv_and_txt(self):
        self.assertEqual(load_file("a.tsv", b"x\ty\n1\t2\n").datasets["data"].shape, (1, 2))
        self.assertEqual(load_file("a.txt", b"x|y\n1|2\n").datasets["data"].shape, (1, 2))

    def test_latin1_encoding_warns(self):
        lf = load_file("a.csv", "name\nJos\xe9\n".encode("latin-1"))
        self.assertTrue(any("not valid UTF-8" in w for w in lf.warnings))
        self.assertEqual(lf.datasets["data"]["name"][0], "José")

    def test_malformed_rows_reported_not_silent(self):
        lf = load_file("a.csv", b"a,b\n1,2\n3,4,5,6\n7,8\n")
        self.assertEqual(len(lf.datasets["data"]), 2)
        self.assertTrue(any("malformed" in w for w in lf.warnings))

    def test_empty_and_invalid(self):
        for name, data in [("a.csv", b""), ("a.exe", b"x"), ("a.csv", b"\n\n")]:
            with self.assertRaises(IngestionError):
                load_file(name, data)
        with self.assertRaises(IngestionError):
            load_file("a.csv", b"x,y\n1,2\n", max_mb=0)
        with self.assertRaises(IngestionError):
            load_file("a.xlsx", b"not an excel file")

    def test_excel_multiple_sheets(self):
        data = excel_bytes({"S1": pd.DataFrame({"a": [1, 2]}), "S2": pd.DataFrame({"b": [3]})})
        lf = load_file("w.xlsx", data)
        self.assertEqual(set(lf.datasets), {"S1", "S2"})
        self.assertEqual(lf.datasets["S1"].shape, (2, 1))

    def test_json_nested_and_lines(self):
        data = json.dumps([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}]).encode()
        self.assertIn("b.c", load_file("a.json", data).datasets["data"].columns)
        self.assertEqual(load_file("a.json", b'{"a":1}\n{"a":2}\n').datasets["data"].shape, (2, 1))
        with self.assertRaises(IngestionError):
            load_file("a.json", b"{broken")

    def test_nested_lists_become_text(self):
        lf = load_file("a.json", json.dumps([{"a": [1, 2]}, {"a": [3]}]).encode())
        self.assertIsInstance(lf.datasets["data"]["a"][0], str)

    def test_duplicate_and_blank_headers_fixed(self):
        lf = load_file("a.csv", b"x,x, \n1,2,3\n")
        cols = list(lf.datasets["data"].columns)
        self.assertEqual(len(set(cols)), 3)
        self.assertTrue(lf.warnings)

    def test_parquet_roundtrip_if_available(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        import io
        buf = io.BytesIO()
        pd.DataFrame({"a": [1, 2]}).to_parquet(buf)
        self.assertEqual(load_file("a.parquet", buf.getvalue()).datasets["data"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()

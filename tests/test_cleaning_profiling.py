import unittest

import numpy as np
import pandas as pd

from tests.helpers import make_session
from analyst import cleaning, profiling
from analyst.filtering import apply_conditions
from analyst.util import AnalystError


def messy():
    return pd.DataFrame({
        "id": range(1, 9), "city": ["Delhi", "delhi ", "Mumbai", "Mumbai", None, "Pune", "Delhi", "Delhi"],
        "amount": [10, 20, None, 40, 50, -5, 1000, 20], "date": ["2024-01-05", "2024-02-10", "2024-02-11", "2024-03-01", "bad", "2024-04-01", "2024-04-02", "2024-02-10"]})


class ProfilingTests(unittest.TestCase):
    def test_kinds(self):
        p = profiling.profile_dataset(messy())
        k = {c["name"]: c["kind"] for c in p["columns_profile"]}
        self.assertEqual(k["id"], "identifier")
        self.assertEqual(k["amount"], "numeric")
        self.assertEqual(k["city"], "categorical")
        self.assertEqual(k["date"], "datetime")  # >=80% parse

    def test_missing_and_duplicates(self):
        df = pd.concat([messy(), messy().iloc[[0]]], ignore_index=True)
        m = profiling.find_missing(df).set_index("column")
        self.assertEqual(m.loc["amount", "missing"], 1)
        n, sample = profiling.find_duplicates(df.drop(columns="id"))
        self.assertGreaterEqual(n, 1)

    def test_issue_detection(self):
        codes = {(i["code"], i["column"]) for i in profiling.detect_issues(messy())}
        self.assertIn(("missing", "amount"), codes)
        self.assertIn(("whitespace", "city"), codes)
        self.assertIn(("label_variants", "city"), codes)
        self.assertIn(("suspicious_values", "amount"), codes)

    def test_quality_score_bounds_and_formula(self):
        q = profiling.profile_dataset(messy())["quality"]
        c = q["components"]
        self.assertTrue(0 <= q["score"] <= 100)
        self.assertAlmostEqual(q["score"], round(.4 * c["completeness"] + .2 * c["uniqueness"] + .2 * c["consistency"] + .2 * c["validity"], 1), 1)

    def test_clean_data_scores_100(self):
        self.assertEqual(profiling.profile_dataset(pd.DataFrame({"a": [1.5, 2.5, 3.5, 9.1], "b": ["x", "y", "x", "y"]}))["quality"]["score"], 100.0)


class CleaningTests(unittest.TestCase):
    def test_original_preserved_and_undo_reset(self):
        s = make_session(messy())
        orig = s.original.copy()
        r = cleaning.run_operation(s, "fill_missing", {"columns": ["amount"], "method": "median"})
        self.assertEqual(r["status"], "applied")
        self.assertEqual(s.working["amount"].isna().sum(), 0)
        pd.testing.assert_frame_equal(s.original, orig)
        self.assertEqual(len(s.log), 1)
        self.assertTrue(s.undo())
        self.assertEqual(s.working["amount"].isna().sum(), 1)
        cleaning.run_operation(s, "trim_text", {})
        s.reset()
        pd.testing.assert_frame_equal(s.working, orig)

    def test_fill_numeric_methods(self):
        s = make_session(messy())
        cleaning.run_operation(s, "fill_missing", {"columns": ["amount"], "method": "mean"})
        self.assertAlmostEqual(s.working.loc[2, "amount"], messy()["amount"].mean())
        s2 = make_session(messy())
        cleaning.run_operation(s2, "fill_missing", {"columns": ["amount"], "method": "value", "value": "7"})
        self.assertEqual(s2.working.loc[2, "amount"], 7)

    def test_fill_categorical_mode_and_errors(self):
        s = make_session(messy())
        cleaning.run_operation(s, "fill_missing", {"columns": ["city"], "method": "mode"})
        self.assertEqual(s.working.loc[4, "city"], "Delhi")
        r = cleaning.run_operation(make_session(messy()), "fill_missing", {"columns": ["city"], "method": "median"})
        self.assertEqual(r["status"], "error")

    def test_duplicates_need_confirmation_then_removed(self):
        df = pd.concat([messy(), messy().iloc[[0]]], ignore_index=True)
        s = make_session(df)
        r = cleaning.run_operation(s, "remove_duplicates", {})
        self.assertEqual(r["status"], "needs_confirmation")
        self.assertEqual(len(s.working), len(df))  # NOT applied
        self.assertIsNotNone(s.pending)
        r = cleaning.run_operation(s, "remove_duplicates", {}, confirmed=True)
        self.assertEqual(r["status"], "applied")
        self.assertEqual(len(s.working), len(df) - 1)

    def test_dry_run_changes_nothing(self):
        s = make_session(messy())
        r = cleaning.run_operation(s, "drop_columns", {"columns": ["id"]}, dry_run=True)
        self.assertEqual(r["status"], "preview")
        self.assertIn("id", s.working.columns)
        self.assertEqual(s.log, [])

    def test_trim_and_standardize(self):
        s = make_session(messy())
        cleaning.run_operation(s, "trim_text", {"columns": ["city"]})
        self.assertNotIn("delhi ", s.working["city"].tolist())
        cleaning.run_operation(s, "standardize_labels", {"column": "city"})
        self.assertEqual(set(s.working["city"].dropna()), {"Delhi", "Mumbai", "Pune"})

    def test_convert_and_date_parsing(self):
        s = make_session(pd.DataFrame({"p": ["1,200", "$30", "x"], "d": ["2024-01-01", "01/02/2024", "2024-03-05"]}))
        r = cleaning.run_operation(s, "convert_type", {"column": "p", "target": "numeric"})
        self.assertEqual(r["status"], "needs_confirmation")  # 'x' would become missing
        cleaning.run_operation(s, "convert_type", {"column": "p", "target": "numeric"}, confirmed=True)
        self.assertEqual(s.working["p"].iloc[0], 1200)
        cleaning.run_operation(s, "parse_dates", {"column": "d"})
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(s.working["d"]))

    def test_outliers_cap_and_flag(self):
        s = make_session(pd.DataFrame({"x": list(range(1, 21)) + [1000]}))
        r = cleaning.run_operation(s, "handle_outliers", {"column": "x", "action": "cap"})
        self.assertEqual(r["status"], "needs_confirmation")
        cleaning.run_operation(s, "handle_outliers", {"column": "x", "action": "cap"}, confirmed=True)
        self.assertLess(s.working["x"].max(), 1000)

    def test_rename_reorder_drop_custom(self):
        s = make_session(messy())
        cleaning.run_operation(s, "rename_columns", {"mapping": {"amount": "amt"}})
        cleaning.run_operation(s, "reorder_columns", {"order": ["amt"]})
        self.assertEqual(s.working.columns[0], "amt")
        cleaning.run_operation(s, "custom_rule", {"column": "amt", "conditions": [{"column": "amt", "op": "<", "value": 0}], "value": "0"}, confirmed=True)
        self.assertEqual(s.working["amt"].min(), 0)
        self.assertEqual(cleaning.run_operation(s, "drop_columns", {"columns": list(s.working.columns)}, confirmed=True)["status"], "error")

    def test_unknown_op_and_bad_params(self):
        s = make_session(messy())
        self.assertEqual(cleaning.run_operation(s, "nope", {})["status"], "error")
        self.assertEqual(cleaning.run_operation(s, "fill_missing", {"bogus": 1})["status"], "error")

    def test_diff_preview(self):
        s = make_session(messy())
        r = cleaning.run_operation(s, "fill_missing", {"columns": ["amount"], "method": "median"}, dry_run=True)["result"]
        d = cleaning.diff_preview(s.working, r.df)
        self.assertEqual(len(d["changed_cells_sample"]), 1)
        self.assertEqual(r.cells_changed, 1)


class FilteringTests(unittest.TestCase):
    def test_operators(self):
        df = messy()
        self.assertEqual(len(apply_conditions(df, [{"column": "city", "op": "==", "value": "DELHI"}])), 4)  # case/whitespace-insensitive
        self.assertEqual(len(apply_conditions(df, [{"column": "amount", "op": "between", "value": [10, 40]}])), 4)
        self.assertEqual(len(apply_conditions(df, [{"column": "city", "op": "in", "value": "Pune, Mumbai"}])), 3)
        self.assertEqual(len(apply_conditions(df, [{"column": "city", "op": "is null"}])), 1)
        self.assertEqual(len(apply_conditions(df, [{"column": "city", "op": "contains", "value": "umb"}])), 2)
        self.assertEqual(len(apply_conditions(df, [{"column": "amount", "op": ">", "value": 30}, {"column": "city", "op": "==", "value": "Delhi"}])), 1)
        self.assertEqual(len(apply_conditions(df, [{"column": "amount", "op": ">", "value": 500}, {"column": "city", "op": "==", "value": "Pune"}], "or")), 2)

    def test_errors(self):
        with self.assertRaises(AnalystError):
            apply_conditions(messy(), [{"column": "nope", "op": "=="}])
        with self.assertRaises(AnalystError):
            apply_conditions(messy(), [{"column": "amount", "op": ">", "value": "abc"}])
        with self.assertRaises(AnalystError):
            apply_conditions(messy(), [{"column": "amount", "op": "like", "value": 1}])

    def test_session_filters_do_not_change_data(self):
        s = make_session(messy())
        s.set_filters([{"column": "city", "op": "==", "value": "Mumbai"}])
        self.assertEqual(len(s.view()), 2)
        self.assertEqual(len(s.working), 8)


if __name__ == "__main__":
    unittest.main()

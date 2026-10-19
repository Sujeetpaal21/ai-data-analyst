import unittest

import numpy as np
import pandas as pd

from tests.helpers import make_df
from analyst import analytics as A
from analyst.util import AnalystError


class StatsTests(unittest.TestCase):
    def test_numeric_stats_match_pandas(self):
        x = pd.Series([1, 2, 3, 4, 100.0])
        s = A.numeric_stats(x)
        self.assertEqual(s["count"], 5)
        self.assertAlmostEqual(s["mean"], x.mean())
        self.assertAlmostEqual(s["median"], 3)
        self.assertAlmostEqual(s["std"], x.std())
        self.assertEqual(s["distribution"], "right-skewed")
        self.assertLess(s["mean_ci95"][0], s["mean"] < s["mean_ci95"][1] and s["mean"])

    def test_inappropriate_columns_skipped(self):
        st = A.calculate_statistics(make_df())
        self.assertIn("order_id", st["skipped"])
        self.assertNotIn("order_id", st["numeric"])
        self.assertIn("city", st["categorical"])
        self.assertIn("order_date", st["datetime"])

    def test_numeric_text_is_handled(self):
        self.assertEqual(A.numeric_stats(pd.Series(["1,000", "2,000"]))["sum"], 3000)


class GroupTests(unittest.TestCase):
    def test_group_sum_matches_pandas(self):
        df = make_df()
        out, info = A.group_and_aggregate(df, ["city"], {"revenue": "sum"})
        exp = df.groupby("city")["revenue"].sum().sort_values(ascending=False)
        self.assertEqual(list(out["city"]), list(exp.index))
        np.testing.assert_allclose(out["sum_revenue"], exp.values)
        self.assertEqual(info["rows_used"], len(df))

    def test_top_n_pct_and_multi_agg(self):
        df = make_df()
        out, _ = A.group_and_aggregate(df, ["product"], {"revenue": ["sum", "mean"], "*": "count"}, top_n=2, pct_of_total=True)
        self.assertEqual(len(out), 2)
        self.assertIn("mean_revenue", out.columns)
        self.assertIn("pct_sum_revenue", out.columns)

    def test_monthly_grouping(self):
        df = make_df()
        out, info = A.group_and_aggregate(df, ["order_date"], {"revenue": "sum"}, freq="month")
        self.assertEqual(info["freq"], "M")
        self.assertAlmostEqual(out["sum_revenue"].sum(), df["revenue"].sum())
        self.assertTrue(out["order_date"].is_monotonic_increasing)

    def test_errors(self):
        df = make_df()
        with self.assertRaises(AnalystError):
            A.group_and_aggregate(df, ["city"], {"revenue": "mode"})
        with self.assertRaises(AnalystError):
            A.group_and_aggregate(df, ["city"], {"product": "sum"})
        with self.assertRaises(AnalystError):
            A.group_and_aggregate(df, ["nope"], {})

    def test_missing_keys_reported(self):
        df = make_df()
        df.loc[:9, "city"] = None
        _, info = A.group_and_aggregate(df, ["city"], {"revenue": "sum"})
        self.assertEqual(info["rows_excluded_missing_keys"], 10)


class CorrelationTests(unittest.TestCase):
    def test_correlation_known(self):
        x = np.arange(100.0)
        df = pd.DataFrame({"a": x, "b": 2 * x + 1, "c": np.random.default_rng(0).normal(size=100)})
        r = A.correlation_analysis(df)
        top = r["pairs"].iloc[0]
        self.assertEqual({top["column_a"], top["column_b"]}, {"a", "b"})
        self.assertAlmostEqual(top["correlation"], 1.0, places=3)
        self.assertEqual(top["n"], 100)

    def test_needs_two_columns(self):
        with self.assertRaises(AnalystError):
            A.correlation_analysis(pd.DataFrame({"a": [1, 2, 3]}))

    def test_target_associations_finds_driver(self):
        df = make_df(600)
        t = A.target_associations(df, "churn")["table"]
        self.assertEqual(t.iloc[0]["feature"], "city")  # churn was built from city
        with self.assertRaises(AnalystError):
            A.target_associations(df, "order_id")


class TimeSeriesTests(unittest.TestCase):
    def test_monthly_series_and_growth(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]), "v": [100, 150, 75]})
        r = A.time_series_analysis(df, "d", "v", "month")
        t = r["table"]
        self.assertEqual(list(t["v".join(["sum_", ""])] if False else t["sum_v"]), [100, 150, 75])
        self.assertAlmostEqual(t["pct_change"].iloc[1], 50.0)
        self.assertAlmostEqual(t["pct_change"].iloc[2], -50.0)
        self.assertAlmostEqual(r["latest_vs_previous"]["pct_change"], -50.0)
        self.assertTrue(r["last_period_may_be_partial"])

    def test_trend_detected(self):
        df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=24, freq="MS"), "v": np.arange(24) * 10.0 + 5})
        r = A.time_series_analysis(df, "d", "v", "month")
        self.assertEqual(r["trend"]["direction"], "increasing")
        self.assertEqual(r["best_period"]["value"], 235.0)

    def test_last_n_and_errors(self):
        r = A.time_series_analysis(make_df(), "order_date", "revenue", "month", last_n_periods=3)
        self.assertEqual(len(r["table"]), 3)
        with self.assertRaises(AnalystError):
            A.time_series_analysis(pd.DataFrame({"d": [None, None], "v": [1, 2]}), "d", "v")

    def test_drivers(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2024-01-05"] * 2 + ["2024-02-05"] * 2), "reg": ["N", "S", "N", "S"], "v": [100, 100, 40, 95]})
        r = A.period_drivers(df, "d", "v", ["reg"], "month")
        top = r["dimensions"]["reg"].iloc[0]
        self.assertEqual(top["category"], "N")
        self.assertEqual(r["total_change"], -65)
        self.assertAlmostEqual(top["share_of_total_change_pct"], 92.3, places=1)


class OutlierKpiQueryTests(unittest.TestCase):
    def test_iqr_outliers(self):
        df = pd.DataFrame({"x": list(range(1, 21)) + [500]})
        r = A.detect_outliers(df, "x")
        self.assertEqual(r["n_outliers"], 1)
        self.assertEqual(r["rows"]["x"].iloc[0], 500)

    def test_zscore_and_isolation_forest(self):
        df = make_df()
        df.loc[0, "revenue"] = 1e6
        self.assertGreaterEqual(A.detect_outliers(df, "revenue", "zscore")["n_outliers"], 1)
        self.assertGreater(A.detect_outliers(df, ["revenue", "price"], "isolation_forest")["n_outliers"], 0)

    def test_kpis(self):
        df = make_df()
        t = A.calculate_kpis(df, [{"column": "revenue", "agg": "sum"}], "order_date", "month")
        self.assertEqual(t.iloc[0]["value"], len(df))
        self.assertAlmostEqual(t.iloc[1]["value"], df["revenue"].sum())
        self.assertIsNotNone(t.iloc[1]["change_pct"])

    def test_query(self):
        out, total = A.query_dataset(make_df(), ["city", "revenue"], [{"column": "city", "op": "==", "value": "Delhi"}], "revenue", False, 5)
        self.assertEqual(len(out), 5)
        self.assertEqual(list(out.columns), ["city", "revenue"])
        self.assertTrue(out["revenue"].is_monotonic_decreasing)
        self.assertGreater(total, 5)


class AdvancedTests(unittest.TestCase):
    def test_model_has_validation_and_no_leakage(self):
        df = make_df(500)
        df["revenue_copy"] = df["revenue"]
        r = A.train_model(df, "revenue")
        self.assertEqual(r["task"], "regression")
        self.assertIn("revenue_copy", r["excluded_features"])
        self.assertIn("order_id", r["excluded_features"])
        self.assertEqual(r["n_train"] + r["n_test"], 500)
        self.assertIn("baseline_mae_(predict_mean)", r["metrics"])
        self.assertGreater(r["metrics"]["r2"], 0.8)

    def test_classification(self):
        r = A.train_model(make_df(500), "churn")
        self.assertEqual(r["task"], "classification")
        self.assertIn("baseline_accuracy_(majority_class)", r["metrics"])
        self.assertIn("estimates", r["note"])

    def test_model_errors(self):
        with self.assertRaises(AnalystError):
            A.train_model(make_df(10), "revenue")

    def test_clustering(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": np.r_[rng.normal(0, .3, 100), rng.normal(10, .3, 100)], "b": np.r_[rng.normal(0, .3, 100), rng.normal(10, .3, 100)]})
        r = A.cluster_data(df)
        self.assertEqual(r["k"], 2)
        self.assertGreater(r["silhouette"], 0.8)

    def test_cohort_and_funnel(self):
        ev = pd.DataFrame({"u": [1, 2, 1, 2, 3], "d": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-02-05", "2024-03-01", "2024-02-01"])})
        r = A.cohort_retention(ev, "u", "d", "month")["retention_pct"]
        self.assertEqual(r.iloc[0]["cohort_size"], 2)
        self.assertEqual(r.iloc[0]["+0"], 100.0)
        self.assertEqual(r.iloc[0]["+1"], 50.0)
        f = pd.DataFrame({"stage": ["visit"] * 4 + ["buy"] * 1, "u": [1, 2, 3, 4, 1]})
        t = A.funnel_analysis(f, "stage", "u", ["visit", "buy"])
        self.assertEqual(list(t["users"]), [4, 1])
        self.assertEqual(t["pct_of_first_stage"].iloc[1], 25.0)


if __name__ == "__main__":
    unittest.main()

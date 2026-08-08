import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_memory import contextualize_question, merge_follow_up_plan
from data_quality import (
    apply_cleaning,
    calculate_health_score,
    cleaning_suggestions,
    profile_data_quality,
)
from insights_engine import generate_business_insights, generate_recommendations


class DataSenseV2Tests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Order Date": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "Region": ["California"] * 12 + ["Texas"] * 12,
            "Category": ["Technology", "Furniture"] * 12,
            "Sales": [100, 120, None, 150, 170, 180, 200, 220, 240, 260, 5000, 300] * 2,
            "Profit": [20, 15, 25, 18, 30, 35, 40, 42, 45, 50, -200, 60] * 2,
            "Discount": [0, .1, .05, .2, .1, .05, 0, .1, .2, .15, .8, .1] * 2,
            "Constant": ["A"] * 24,
        })
        self.df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

    def test_quality_report_and_score(self):
        report = profile_data_quality(self.df)
        score = calculate_health_score(self.df, report)
        self.assertEqual(report["missing_columns"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["constant_columns"], ["Constant"])
        self.assertFalse(report["outliers"].empty)
        self.assertTrue(0 <= score["Overall"] <= 100)

    def test_user_controlled_cleaning(self):
        report = profile_data_quality(self.df)
        suggestions = cleaning_suggestions(self.df, report)
        self.assertEqual(suggestions["Sales"][0], "Fill with median")
        cleaned, log = apply_cleaning(
            self.df,
            {"Sales": "Fill with median"},
            remove_duplicates=True,
            outlier_action="Cap at IQR bounds",
            report=report,
        )
        self.assertEqual(cleaned["Sales"].isna().sum(), 0)
        self.assertEqual(cleaned.duplicated().sum(), 0)
        self.assertTrue(log)

    def test_grounded_insights_and_recommendations(self):
        insights = generate_business_insights(self.df)
        recommendations = generate_recommendations(insights)
        self.assertTrue(insights)
        self.assertTrue(recommendations)
        self.assertFalse(any("A has the highest" in item["text"] for item in insights))

    def test_follow_up_memory(self):
        previous = {
            "measure": "Sales",
            "operation": "sum",
            "analysis_type": "aggregation",
            "chart": "bar",
            "group_by": ["Region"],
            "filters": [],
        }
        context, follow_up = contextualize_question("Only for 2023", previous)
        self.assertTrue(follow_up)
        self.assertIn("Sales", context)
        self.assertIn("Region", context)

        merged = merge_follow_up_plan(
            {"measure": None, "operation": None, "analysis_type": None,
             "chart": None, "group_by": [], "filters": []},
            previous,
            "Only for 2023",
        )
        self.assertEqual(merged["measure"], "Sales")
        self.assertEqual(merged["group_by"], ["Region"])
        self.assertFalse(contextualize_question("Show profit by category", previous)[1])


if __name__ == "__main__":
    unittest.main()

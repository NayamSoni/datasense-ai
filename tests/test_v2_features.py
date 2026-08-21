import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_memory import contextualize_question, merge_follow_up_plan
from data_quality import (
    apply_cleaning,
    calculate_health_score,
    cleaning_suggestions,
    profile_data_quality,
)
from insights_engine import (
    generate_business_insights,
    generate_decision_report,
    generate_recommendations,
    report_to_markdown,
)
from pandas_agent import build_calculation_audit, calculate
from query_planner import apply_rule_engine, create_execution_plan, resolve_plan


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
        self.assertTrue(all(item.get("summary") for item in insights))
        self.assertTrue(all(len(item["summary"]) <= 180 for item in insights))
        self.assertTrue(all(len(item["text"]) >= len(item["summary"]) for item in insights))

    def test_decision_report_explains_volume_and_value_drivers(self):
        monthly = pd.DataFrame({
            "Admission Date": (
                [pd.Timestamp("2024-03-15")] * 100
                + [pd.Timestamp("2024-04-15")] * 120
            ),
            "Admission Type": ["Urgent", "Emergency"] * 110,
            "Billing Amount": [100.0] * 100 + [90.0] * 120,
            "Room Number": list(range(220)),
        })

        report = generate_decision_report(monthly)
        movement = next(
            item for item in report["findings"]
            if item["type"] == "growth"
        )

        self.assertIn("volume-led", movement["meaning"])
        self.assertIn("average Billing Amount per record", movement["meaning"])
        self.assertNotIn("Room Number", report["overview"]["metrics"])
        self.assertTrue(any("No reliable entity identifier" in item for item in report["limitations"]))

    def test_balanced_segments_do_not_create_concentration_risk(self):
        balanced = pd.DataFrame({
            "Gender": ["Female"] * 100 + ["Male"] * 100,
            "Billing Amount": [100.0] * 100 + [100.5] * 100,
        })

        report = generate_decision_report(balanced)
        finding_types = {item["type"] for item in report["findings"]}

        self.assertIn("balanced_segments", finding_types)
        self.assertNotIn("concentration", finding_types)
        self.assertFalse(any("concentration risk" in item["action"].lower() for item in report["recommendations"]))

    def test_material_concentration_is_reported(self):
        concentrated = pd.DataFrame({
            "Segment": ["A"] * 80 + ["B"] * 20,
            "Sales": [100.0] * 100,
        })

        report = generate_decision_report(concentrated)
        concentration = next(
            item for item in report["findings"]
            if item["type"] == "concentration"
        )

        self.assertIn("80.0%", concentration["evidence"])
        self.assertEqual(concentration["confidence"], "High")

    def test_calendar_length_is_not_presented_as_proven_seasonality(self):
        rows = []
        for date in pd.date_range("2019-01-01", "2023-12-31", freq="D"):
            rows.append({
                "Date": date,
                "Category": "All",
                "Sales": 1.04 if date.month == 7 else 1.0,
            })
        calendar_data = pd.DataFrame(rows)

        report = generate_decision_report(calendar_data)
        markdown = report_to_markdown(report, "Calendar Test")

        self.assertTrue(any(item["type"] == "calendar_effect" for item in report["findings"]))
        self.assertNotIn("Prepare capacity, inventory, and campaigns", markdown)
        self.assertIn("Limitations and Trust Notes", markdown)

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

    def test_highest_entity_returns_name_value_filter_and_audit(self):
        healthcare = pd.DataFrame({
            "Hospital Name": ["Apollo", "Apollo", "Fortis", "Fortis", "Manipal"],
            "Billing Amount": [100, 200, 250, 100, 500],
            "Insurance Provider": [
                "Medicare",
                "Medicare",
                "Medicare",
                "Private",
                "Medicare",
            ],
        })
        question = "Which hospital has the highest billing amount for Medicare patients?"
        raw_plan = {
            "title": question,
            "analysis_type": "aggregation",
            "operation": "max",
            "measure": "Billing Amount",
            "group_by": ["Hospital Name"],
            "filters": [],
            "sort": None,
            "limit": None,
            "chart": "table",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertEqual(plan["analysis_type"], "top_bottom")
        self.assertEqual(plan["operation"], "sum")
        self.assertEqual(plan["group_by"], ["Hospital Name"])
        self.assertEqual(
            plan["filters"],
            [{"column": "Insurance Provider", "value": "Medicare"}],
        )
        self.assertEqual(result.iloc[0]["Hospital Name"], "Manipal")
        self.assertEqual(result.iloc[0]["Billing Amount"], 500)
        self.assertEqual(audit["valid_measure_rows"], 4)
        self.assertEqual(audit["groups_evaluated"], 3)
        self.assertIn("Hospital Name", audit["formula"])

    def test_urgent_patient_question_counts_distinct_ids_with_audit(self):
        healthcare = pd.DataFrame({
            "Patient ID": ["P1", "P2", "P2", "P3", "P4"],
            "Admission Type": [
                "Urgent",
                "Urgent",
                "Urgent",
                "Emergency",
                "Elective",
            ],
            "Billing Amount": [100, 200, 300, 400, 500],
        })
        question = "What is the total number of patients admitted in Urgent category?"
        raw_plan = {
            "title": question,
            "analysis_type": "aggregation",
            "operation": "sum",
            "measure": "Admission Type",
            "group_by": ["Admission Type"],
            "filters": [],
            "sort": None,
            "limit": None,
            "chart": "table",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertEqual(plan["analysis_type"], "patient_count")
        self.assertEqual(plan["patient_column"], "Patient ID")
        self.assertEqual(
            plan["filters"],
            [{"column": "Admission Type", "value": "Urgent"}],
        )
        self.assertEqual(result.iloc[0]["Distinct Patients"], 2)
        self.assertEqual(audit["rows_after_filters"], 3)
        self.assertEqual(audit["valid_measure_rows"], 3)
        self.assertEqual(audit["formula"], "COUNT(DISTINCT Patient ID)")
        self.assertEqual(audit["count_basis"], "Distinct Patient ID")

    def test_patient_count_without_id_is_labeled_as_admission_records(self):
        healthcare = pd.DataFrame({
            "Admission Type": ["Urgent", "Urgent", "Emergency", "Urgent"],
            "Billing Amount": [100, 200, 300, 400],
        })
        question = "How many patients were admitted in Urgent category?"
        raw_plan = {
            "title": question,
            "analysis_type": "aggregation",
            "operation": "sum",
            "measure": "Admission Type",
            "group_by": [],
            "filters": [],
            "chart": "table",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertIsNone(plan["patient_column"])
        self.assertEqual(result.iloc[0]["Patient Admission Records"], 3)
        self.assertEqual(audit["formula"], "COUNT(admission rows)")
        self.assertIn("no patient identifier", audit["count_basis"])

    def test_informal_age_histogram_uses_age_not_default_amount(self):
        healthcare = pd.DataFrame({
            "Age": [13, 18, 21, 29, 35, 44, 51, 67, 72, 89],
            "Billing Amount": [1000, 1800, 2100, 2900, 3500,
                               4400, 5100, 6700, 7200, 8900],
            "Admission Type": ["Urgent", "Emergency"] * 5,
        })
        question = (
            "I want to see a histogram of how many patients are "
            "in each age bucket."
        )
        # Simulate a small local model returning the same stale Billing Amount
        # measure that caused the incorrect chart in the UI.
        raw_plan = {
            "title": question,
            "analysis_type": "aggregation",
            "operation": "sum",
            "measure": "Billing Amount",
            "group_by": [],
            "filters": [],
            "chart": "bar",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertEqual(plan["analysis_type"], "distribution")
        self.assertEqual(plan["measure"], "Age")
        self.assertEqual(plan["operation"], "count")
        self.assertEqual(plan["chart"], "histogram")
        self.assertEqual(plan["title"], "Age Distribution")
        self.assertEqual(result["Range"].tolist()[0], "10–19")
        self.assertEqual(result["Range"].tolist()[-1], "80–89")
        self.assertEqual(int(result["Count"].sum()), len(healthcare))
        self.assertEqual(audit["groups_evaluated"], 8)
        self.assertEqual(
            audit["formula"],
            "COUNT(records) grouped into Age buckets",
        )
        self.assertEqual(
            audit["count_basis"],
            "Rows with a non-null numeric Age",
        )

    def test_informal_distribution_phrasings_resolve_from_dataset_schema(self):
        healthcare = pd.DataFrame({
            "Age": [18, 25, 44],
            "Billing Amount": [100, 200, 300],
        })
        questions = (
            "Show me the distribution of ages",
            "Can you group the ages and show how many fall in each range?",
            "Plot an age-wise frequency chart",
        )

        for question in questions:
            with self.subTest(question=question):
                raw_plan = {
                    "title": question,
                    "analysis_type": "aggregation",
                    "operation": "sum",
                    "measure": "Billing Amount",
                    "group_by": [],
                    "filters": [],
                    "chart": "table",
                }
                plan = resolve_plan(
                    healthcare,
                    apply_rule_engine(question, raw_plan),
                    question,
                )
                self.assertEqual(plan["analysis_type"], "distribution")
                self.assertEqual(plan["measure"], "Age")
                self.assertEqual(plan["chart"], "histogram")

    def test_distribution_fallback_works_when_local_llm_returns_no_plan(self):
        healthcare = pd.DataFrame({
            "Age": [18, 25, 44],
            "Billing Amount": [100, 200, 300],
        })
        question = "How many patients are in each age bucket?"

        with patch("query_planner.llm_plan", return_value=[]):
            plans = create_execution_plan(question, healthcare)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["analysis_type"], "distribution")
        self.assertEqual(plans[0]["measure"], "Age")
        self.assertEqual(plans[0]["chart"], "histogram")

    def test_numeric_to_category_correlation_becomes_group_relationship(self):
        healthcare = pd.DataFrame({
            "Admission Type": [
                "Elective", "Elective", "Emergency",
                "Emergency", "Urgent", "Urgent",
            ],
            "Billing Amount": [100.0, 110.0, 100.0, 110.0, 100.0, 110.0],
            "Age": [20, 30, 40, 50, 60, 70],
        })
        question = "Correlation between Billing Amount and Admission Type"
        raw_plan = {
            "title": question,
            "analysis_type": "correlation",
            "operation": "mean",
            "measure": "Billing Amount",
            "measure2": "Admission Type",
            "group_by": [],
            "filters": [],
            "chart": "scatter",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertEqual(plan["analysis_type"], "categorical_relationship")
        self.assertEqual(plan["measure"], "Billing Amount")
        self.assertIsNone(plan["measure2"])
        self.assertEqual(plan["group_by"], ["Admission Type"])
        self.assertEqual(plan["chart"], "box")
        self.assertEqual(plan["title"], "Billing Amount by Admission Type")
        self.assertEqual(result["Records"].sum(), len(healthcare))
        self.assertEqual(set(result["Admission Type"]), {
            "Elective", "Emergency", "Urgent",
        })
        self.assertIn("Average Billing Amount", result.columns)
        self.assertIn("Association (η²)", result.columns)
        self.assertEqual(result["Association Strength"].iloc[0], "Negligible")
        self.assertEqual(audit["groups_evaluated"], 3)
        self.assertEqual(audit["valid_measure_rows"], 6)
        self.assertIn("ETA_SQUARED", audit["formula"])

    def test_two_numeric_columns_still_use_pearson_correlation(self):
        healthcare = pd.DataFrame({
            "Age": [20, 30, 40, 50, 60],
            "Billing Amount": [100, 200, 300, 400, 500],
            "Admission Type": ["Urgent"] * 5,
        })
        question = "Correlation between Age and Billing Amount"
        raw_plan = {
            "title": question,
            "analysis_type": "correlation",
            "operation": "mean",
            "measure": "Age",
            "measure2": "Billing Amount",
            "group_by": [],
            "filters": [],
            "chart": "scatter",
        }

        plan = resolve_plan(
            healthcare,
            apply_rule_engine(question, raw_plan),
            question,
        )
        result = calculate(healthcare, plan)

        self.assertEqual(plan["analysis_type"], "correlation")
        self.assertEqual(plan["measure"], "Age")
        self.assertEqual(plan["measure2"], "Billing Amount")
        self.assertEqual(result.iloc[0]["Correlation"], 1.0)

    def test_sum_on_text_measure_is_rejected(self):
        healthcare = pd.DataFrame({
            "Admission Type": ["Urgent", "Emergency", "Elective"],
        })
        plan = {
            "analysis_type": "aggregation",
            "operation": "sum",
            "measure": "Admission Type",
            "group_by": [],
            "filters": [],
        }

        with self.assertRaisesRegex(ValueError, "requires a numeric measure"):
            calculate(healthcare, plan)

    def test_total_billing_replaces_invented_filter_with_real_category(self):
        healthcare = pd.DataFrame({
            "Admission Type": ["Urgent", "Urgent", "Emergency", "Elective"],
            "Billing Amount": [100.25, 200.25, 300.00, 400.00],
        })
        question = "What is the total billing amount for all urgent cases?"
        raw_plan = {
            "title": question,
            "analysis_type": "aggregation",
            "operation": "sum",
            "measure": "Billing Amount",
            "group_by": [],
            # Simulate the local model expanding a real value into a value
            # that does not exist in the uploaded dataset.
            "filters": [
                {"column": "Admission Type", "value": "Urgent cases"},
            ],
            "chart": "table",
        }

        plan = resolve_plan(healthcare, raw_plan, question)
        result = calculate(healthcare, plan)
        audit = build_calculation_audit(healthcare, plan)

        self.assertEqual(
            plan["filters"],
            [{"column": "Admission Type", "value": "Urgent"}],
        )
        self.assertEqual(result.iloc[0]["Billing Amount"], 300)
        self.assertEqual(audit["rows_after_filters"], 2)
        self.assertEqual(audit["formula"], "SUM(Billing Amount)")


if __name__ == "__main__":
    unittest.main()

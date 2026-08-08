"""Beginner-friendly predictive-modelling workflow for DataSense AI.

The module keeps machine-learning code outside ``app.py`` so the Streamlit
page router stays readable.  It intentionally starts with one transparent
baseline per problem type:

* Logistic Regression for classification
* Linear Regression for regression

The LLM is not asked to calculate model metrics.  Scikit-learn trains the
model, Python calculates the metrics, and Streamlit presents the results.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_fscore_support,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = ""
except ImportError as exc:  # Keep the rest of DataSense AI usable.
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = str(exc)


MINIMUM_TRAINING_ROWS = 50
MAX_CLASS_COUNT = 50
MAX_CATEGORICAL_LEVELS = 100


def infer_problem_type(target: pd.Series) -> str:
    """Infer classification or regression from the selected target column."""
    non_null = target.dropna()
    unique_count = int(non_null.nunique())

    if (
        is_bool_dtype(non_null)
        or is_object_dtype(non_null)
        or is_string_dtype(non_null)
        or str(non_null.dtype) == "category"
    ):
        return "Classification"

    # Integer targets such as 0/1 cancellation flags are classifications.
    if is_integer_dtype(non_null) and unique_count <= 20:
        return "Classification"

    return "Regression"


def recommend_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[list[str], dict[str, str]]:
    """Recommend safe starter features and explain which fields were excluded."""
    recommended: list[str] = []
    excluded: dict[str, str] = {}
    row_count = max(len(df), 1)

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / row_count

        if unique_count <= 1:
            excluded[column] = "constant or empty"
        elif is_datetime64_any_dtype(series):
            excluded[column] = "date feature engineering will be added later"
        elif (
            (is_object_dtype(series) or is_string_dtype(series))
            and (
                unique_count > MAX_CATEGORICAL_LEVELS
                or unique_ratio > 0.50
            )
        ):
            excluded[column] = "likely identifier or high-cardinality text"
        else:
            recommended.append(column)

    return recommended, excluded


def _readiness_checks(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    problem_type: str,
) -> tuple[list[str], list[str], int]:
    """Return blocking issues, warnings, and usable row count."""
    target = df[target_column]
    usable_rows = int(target.notna().sum())
    unique_targets = int(target.dropna().nunique())
    blocking: list[str] = []
    warnings: list[str] = []

    if usable_rows < MINIMUM_TRAINING_ROWS:
        blocking.append(
            f"At least {MINIMUM_TRAINING_ROWS} rows with a known target are "
            f"required; this dataset has {usable_rows}."
        )

    if not feature_columns:
        blocking.append("Select at least one input feature.")

    if unique_targets < 2:
        blocking.append("The target must contain at least two different values.")

    if problem_type == "Classification":
        if unique_targets > MAX_CLASS_COUNT:
            blocking.append(
                f"The target has {unique_targets} classes. This first version "
                f"supports up to {MAX_CLASS_COUNT}."
            )
        class_counts = target.dropna().value_counts()
        if not class_counts.empty and int(class_counts.min()) < 2:
            warnings.append(
                "At least one class has only one row, so the train/test split "
                "cannot preserve the class distribution."
            )
    elif not is_numeric_dtype(target):
        blocking.append("A regression target must be numeric.")

    missing_feature_cells = int(df[feature_columns].isna().sum().sum()) if feature_columns else 0
    if missing_feature_cells:
        warnings.append(
            f"{missing_feature_cells:,} missing feature values will be filled "
            "inside the modelling pipeline."
        )

    return blocking, warnings, usable_rows


def _build_pipeline(
    x: pd.DataFrame,
    problem_type: str,
) -> tuple[Any, str]:
    """Create preprocessing and a transparent baseline model."""
    numeric_columns = x.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in x.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=2,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if problem_type == "Classification":
        model = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        )
        model_name = "Logistic Regression"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, model_name


def _train_baseline(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    problem_type: str,
    test_percentage: int,
) -> dict[str, Any]:
    """Train on one split and return only verified evaluation outputs."""
    modelling_df = df[feature_columns + [target_column]].copy()
    modelling_df = modelling_df[modelling_df[target_column].notna()]

    x = modelling_df[feature_columns]
    y = modelling_df[target_column]
    test_size = test_percentage / 100

    stratify = None
    split_note = ""
    if problem_type == "Classification":
        class_counts = y.value_counts()
        expected_test_rows = math.ceil(len(y) * test_size)
        if (
            not class_counts.empty
            and int(class_counts.min()) >= 2
            and expected_test_rows >= int(y.nunique())
        ):
            stratify = y
        else:
            split_note = (
                "The class distribution could not be preserved in the split "
                "because one or more classes had too few rows."
            )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline, model_name = _build_pipeline(x_train, problem_type)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    prediction_table = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(predictions),
        }
    )

    result: dict[str, Any] = {
        "model_name": model_name,
        "problem_type": problem_type,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "split_note": split_note,
        "predictions": prediction_table,
    }

    if problem_type == "Classification":
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
        labels = sorted(y.dropna().unique().tolist(), key=str)
        result["metrics"] = {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1),
        }
        result["confusion_matrix"] = pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=labels),
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )
    else:
        mse = mean_squared_error(y_test, predictions)
        result["metrics"] = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(math.sqrt(mse)),
            "R²": float(r2_score(y_test, predictions)),
        }

    return result


def _configuration_id(
    dataset_id: str,
    target_column: str,
    feature_columns: list[str],
    problem_type: str,
    test_percentage: int,
) -> str:
    """Identify the exact settings used for a stored modelling result."""
    raw = "|".join(
        [
            dataset_id,
            target_column,
            *feature_columns,
            problem_type,
            str(test_percentage),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _render_verified_interpretation(result: dict[str, Any]) -> None:
    """Explain calculated metrics without asking an LLM to invent meaning."""
    metrics = result["metrics"]
    if result["problem_type"] == "Classification":
        st.markdown(
            f"""
            **How to read this baseline**

            - The model correctly classified **{metrics['Accuracy']:.1%}** of test rows.
            - The weighted F1 score is **{metrics['F1 Score']:.3f}** and balances
              precision with recall across classes.
            - Use the confusion matrix to see which outcomes the model mixes up.
            """
        )
    else:
        st.markdown(
            f"""
            **How to read this baseline**

            - Predictions are off by **{metrics['MAE']:,.2f}** target units on
              average.
            - Larger errors are reflected in an RMSE of
              **{metrics['RMSE']:,.2f}**.
            - R² is **{metrics['R²']:.3f}**; compare this with future models,
              rather than treating it as proof that the model is production-ready.
            """
        )

    st.caption(
        "These metrics describe one fixed test split. They are a starting "
        "benchmark, not a production-readiness claim."
    )


def render_data_science_lab_intro() -> None:
    """Render the Data Science Lab hero before a dataset is uploaded."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 82% 9%, rgba(0, 153, 255, 0.16), transparent 30%),
                radial-gradient(circle at 18% 45%, rgba(0, 83, 255, 0.11), transparent 36%),
                linear-gradient(145deg, #050b18 0%, #07152b 48%, #061126 100%) !important;
        }
        .ds-lab-shell {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 300px;
            align-items: center;
            gap: 28px;
            overflow: hidden;
            padding: 34px 38px;
            margin: 2px 0 24px;
            border: 1px solid rgba(65, 168, 255, 0.38);
            border-radius: 22px;
            background:
                linear-gradient(120deg, rgba(8, 30, 68, 0.96), rgba(5, 18, 42, 0.90)),
                repeating-linear-gradient(
                    90deg,
                    rgba(255,255,255,0.025) 0,
                    rgba(255,255,255,0.025) 1px,
                    transparent 1px,
                    transparent 34px
                );
            box-shadow:
                0 24px 70px rgba(0, 35, 110, 0.32),
                inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .ds-lab-content {
            position: relative;
            z-index: 2;
            min-width: 0;
        }
        .ds-lab-shell::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -54px;
            top: -82px;
            border-radius: 50%;
            border: 1px solid rgba(57, 190, 255, 0.35);
            box-shadow:
                0 0 0 30px rgba(34, 121, 255, 0.06),
                0 0 0 62px rgba(34, 121, 255, 0.035),
                0 0 80px rgba(0, 174, 255, 0.32);
            animation: ds-lab-pulse 4.5s ease-in-out infinite;
        }
        .ds-lab-kicker {
            color: #54c8ff;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .ds-lab-title {
            position: relative;
            z-index: 1;
            margin: 8px 0 6px;
            color: #f6fbff;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.08;
            font-weight: 760;
            letter-spacing: -0.035em;
        }
        .ds-lab-title span {
            color: #45bdff;
            text-shadow: 0 0 30px rgba(47, 178, 255, 0.45);
        }
        .ds-lab-copy {
            position: relative;
            z-index: 1;
            max-width: 650px;
            color: #adbed7;
            font-size: 0.98rem;
            line-height: 1.65;
        }
        .ds-lab-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 18px;
            padding: 7px 12px;
            border: 1px solid rgba(51, 196, 255, 0.34);
            border-radius: 999px;
            color: #cceeff;
            background: rgba(0, 127, 255, 0.10);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .ds-lab-status::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #39d3ff;
            box-shadow: 0 0 12px #39d3ff;
        }
        .ds-avatar-stage {
            position: relative;
            z-index: 2;
            display: grid;
            place-items: center;
            min-height: 250px;
            isolation: isolate;
        }
        .ds-avatar-halo {
            position: absolute;
            width: 238px;
            height: 238px;
            border: 1px solid rgba(85, 208, 255, 0.42);
            border-radius: 50%;
            background:
                radial-gradient(
                    circle,
                    rgba(111, 80, 255, 0.24) 0%,
                    rgba(20, 139, 255, 0.12) 43%,
                    transparent 70%
                );
            box-shadow:
                0 0 55px rgba(27, 139, 255, 0.30),
                inset 0 0 38px rgba(104, 72, 255, 0.20);
            animation: ds-avatar-pulse 4s ease-in-out infinite;
        }
        .ds-avatar-ring {
            position: absolute;
            width: 270px;
            height: 92px;
            border: 1px solid rgba(83, 205, 255, 0.48);
            border-radius: 50%;
            transform: rotate(-10deg);
            box-shadow: 0 0 24px rgba(31, 163, 255, 0.25);
            animation: ds-avatar-ring 8s linear infinite;
        }
        .ds-avatar-head {
            position: relative;
            width: 166px;
            height: 178px;
            border: 2px solid rgba(157, 220, 255, 0.82);
            border-radius: 48% 48% 43% 43% / 42% 42% 54% 54%;
            background:
                linear-gradient(
                    145deg,
                    rgba(226, 239, 255, 0.96),
                    rgba(89, 110, 174, 0.95) 48%,
                    rgba(35, 24, 91, 0.98)
                );
            box-shadow:
                0 22px 42px rgba(0, 12, 48, 0.48),
                0 0 34px rgba(75, 91, 255, 0.34),
                inset 9px 8px 20px rgba(255,255,255,0.35);
            animation: ds-avatar-float 3.7s ease-in-out infinite;
        }
        .ds-avatar-head::before {
            content: "";
            position: absolute;
            left: 18px;
            right: 18px;
            top: 35px;
            height: 88px;
            border: 1px solid rgba(131, 112, 255, 0.90);
            border-radius: 44% 44% 48% 48%;
            background:
                radial-gradient(circle at 30% 46%, rgba(125, 89, 255, 0.28), transparent 24%),
                radial-gradient(circle at 70% 46%, rgba(67, 192, 255, 0.25), transparent 24%),
                linear-gradient(145deg, #070c2b, #1a1456 55%, #071835);
            box-shadow:
                inset 0 0 26px rgba(101, 67, 255, 0.52),
                0 0 18px rgba(64, 145, 255, 0.38);
        }
        .ds-avatar-antenna {
            position: absolute;
            top: -31px;
            left: 50%;
            width: 2px;
            height: 32px;
            transform: translateX(-50%);
            background: linear-gradient(#70e4ff, #9a7cff);
            box-shadow: 0 0 8px #70e4ff;
        }
        .ds-avatar-antenna::before {
            content: "";
            position: absolute;
            left: 50%;
            top: -8px;
            width: 12px;
            height: 12px;
            transform: translateX(-50%);
            border: 2px solid #dff9ff;
            border-radius: 50%;
            background: #3bcfff;
            box-shadow: 0 0 16px #55d8ff;
        }
        .ds-avatar-ear {
            position: absolute;
            top: 67px;
            width: 19px;
            height: 48px;
            border: 1px solid rgba(134, 210, 255, 0.72);
            background: linear-gradient(#4e73c3, #25215f);
            box-shadow: 0 0 15px rgba(86, 116, 255, 0.38);
        }
        .ds-avatar-ear.left {
            left: -15px;
            border-radius: 12px 4px 4px 12px;
        }
        .ds-avatar-ear.right {
            right: -15px;
            border-radius: 4px 12px 12px 4px;
        }
        .ds-avatar-eye {
            position: absolute;
            z-index: 1;
            top: 75px;
            width: 19px;
            height: 13px;
            border-radius: 50%;
            background: #b992ff;
            box-shadow:
                0 0 8px #b273ff,
                0 0 18px #714eff,
                0 0 28px rgba(66, 186, 255, 0.86);
            animation: ds-avatar-blink 5.5s infinite;
        }
        .ds-avatar-eye.left { left: 49px; }
        .ds-avatar-eye.right { right: 49px; }
        .ds-avatar-mouth {
            position: absolute;
            z-index: 1;
            left: 50%;
            top: 105px;
            width: 30px;
            height: 9px;
            transform: translateX(-50%);
            border-bottom: 3px solid #61d9ff;
            border-radius: 0 0 20px 20px;
            filter: drop-shadow(0 0 6px #5ccfff);
        }
        .ds-avatar-badge {
            position: absolute;
            bottom: -1px;
            padding: 5px 11px;
            border: 1px solid rgba(92, 206, 255, 0.45);
            border-radius: 999px;
            color: #cbedff;
            background: rgba(5, 22, 58, 0.86);
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            box-shadow: 0 8px 20px rgba(0, 13, 50, 0.36);
        }
        .ds-capability-heading {
            margin: 8px 0 8px;
            color: #f2f8ff;
            font-size: 1.15rem;
            font-weight: 720;
        }
        .ds-capability-sub {
            margin: 0 0 15px;
            color: #8298b7;
            font-size: 0.86rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-model-card) {
            min-height: 338px;
            padding: 14px 14px 13px;
            overflow: hidden;
            border: 1px solid rgba(54, 137, 255, 0.30) !important;
            border-radius: 18px !important;
            background:
                linear-gradient(165deg, rgba(18, 63, 137, 0.92), rgba(6, 25, 65, 0.98)) !important;
            box-shadow:
                0 20px 42px rgba(0, 25, 78, 0.28),
                inset 0 1px 0 rgba(255,255,255,0.07);
            transition:
                transform 180ms ease,
                border-color 180ms ease,
                box-shadow 180ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-model-card):hover {
            transform: translateY(-5px);
            border-color: rgba(66, 191, 255, 0.72) !important;
            box-shadow:
                0 25px 54px rgba(0, 77, 175, 0.30),
                inset 0 1px 0 rgba(255,255,255,0.08);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-model-card.is-active) {
            border-color: #4bc9ff !important;
            background:
                linear-gradient(160deg, rgba(24, 91, 192, 0.96), rgba(7, 35, 86, 0.98)) !important;
            box-shadow:
                0 24px 58px rgba(0, 108, 255, 0.32),
                0 0 0 1px rgba(76, 203, 255, 0.14),
                inset 0 1px 0 rgba(255,255,255,0.10);
        }
        .ds-model-card {
            text-align: center;
        }
        .ds-model-art {
            position: relative;
            display: grid;
            place-items: center;
            height: 126px;
            margin-bottom: 16px;
            overflow: hidden;
            border: 1px solid rgba(104, 198, 255, 0.22);
            border-radius: 14px;
            background:
                radial-gradient(circle at 50% 35%, rgba(56, 193, 255, 0.30), transparent 38%),
                linear-gradient(145deg, rgba(34, 94, 202, 0.72), rgba(20, 33, 105, 0.74));
        }
        .ds-model-art::before,
        .ds-model-art::after {
            content: "";
            position: absolute;
            width: 78px;
            height: 78px;
            border: 1px solid rgba(100, 215, 255, 0.24);
            transform: rotate(45deg);
        }
        .ds-model-art::before {
            left: -34px;
            bottom: -42px;
        }
        .ds-model-art::after {
            right: -38px;
            top: -40px;
        }
        .ds-model-card.forecasting .ds-model-art {
            background:
                radial-gradient(circle at 50% 38%, rgba(255, 220, 75, 0.29), transparent 34%),
                linear-gradient(145deg, rgba(38, 101, 213, 0.74), rgba(24, 38, 112, 0.76));
        }
        .ds-model-card.hypothesis .ds-model-art {
            background:
                radial-gradient(circle at 50% 38%, rgba(90, 255, 201, 0.26), transparent 34%),
                linear-gradient(145deg, rgba(32, 99, 201, 0.74), rgba(25, 36, 108, 0.77));
        }
        .ds-model-icon {
            position: relative;
            z-index: 1;
            font-size: 3.3rem;
            line-height: 1;
            filter: drop-shadow(0 12px 18px rgba(0, 10, 45, 0.42));
            animation: ds-model-float 3.6s ease-in-out infinite;
        }
        .ds-model-status {
            display: inline-block;
            margin-bottom: 9px;
            padding: 4px 9px;
            border: 1px solid rgba(92, 203, 255, 0.34);
            border-radius: 999px;
            color: #75d8ff;
            background: rgba(0, 141, 255, 0.11);
            font-size: 0.64rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .ds-model-title {
            margin-bottom: 7px;
            color: #f5f9ff;
            font-size: 1.12rem;
            font-weight: 760;
            letter-spacing: -0.015em;
        }
        .ds-model-copy {
            min-height: 48px;
            color: #a9bbd6;
            font-size: 0.78rem;
            line-height: 1.48;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-model-card)
        .stButton > button {
            min-height: 36px;
            border: 1px solid rgba(84, 196, 255, 0.48) !important;
            border-radius: 8px !important;
            background: rgba(3, 21, 58, 0.48) !important;
            color: #dff5ff !important;
            font-size: 0.72rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.07em !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ds-model-card)
        .stButton > button:hover {
            background: linear-gradient(90deg, #1664e8, #009bd9) !important;
            border-color: #67dcff !important;
        }
        div[data-testid="stMetric"] {
            padding: 15px 16px;
            border: 1px solid rgba(50, 151, 255, 0.28);
            border-radius: 14px;
            background: linear-gradient(
                145deg,
                rgba(9, 40, 86, 0.76),
                rgba(5, 24, 52, 0.84)
            );
        }
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"] {
            border-color: rgba(53, 158, 255, 0.25) !important;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button {
            border: 1px solid rgba(82, 204, 255, 0.58) !important;
            background: linear-gradient(90deg, #135be2, #0099d8) !important;
            color: white !important;
            box-shadow: 0 10px 28px rgba(0, 115, 255, 0.24);
        }
        .ds-coming-soon {
            padding: 26px;
            border: 1px solid rgba(47, 159, 255, 0.30);
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(9, 41, 88, 0.80), rgba(6, 22, 48, 0.90));
        }
        .ds-coming-soon strong {
            color: #58c8ff;
            font-size: 1.05rem;
        }
        .ds-coming-soon p {
            margin: 8px 0 0;
            color: #9fb3ce;
        }
        @keyframes ds-lab-pulse {
            0%, 100% { transform: scale(0.96); opacity: 0.72; }
            50% { transform: scale(1.04); opacity: 1; }
        }
        @keyframes ds-avatar-pulse {
            0%, 100% { transform: scale(0.96); opacity: 0.72; }
            50% { transform: scale(1.04); opacity: 1; }
        }
        @keyframes ds-avatar-ring {
            from { transform: rotate(-10deg); }
            to { transform: rotate(350deg); }
        }
        @keyframes ds-avatar-float {
            0%, 100% { transform: translateY(4px) rotate(-1deg); }
            50% { transform: translateY(-8px) rotate(1deg); }
        }
        @keyframes ds-avatar-blink {
            0%, 44%, 48%, 100% { transform: scaleY(1); }
            46% { transform: scaleY(0.12); }
        }
        @keyframes ds-model-float {
            0%, 100% { transform: translateY(2px); }
            50% { transform: translateY(-7px); }
        }
        @media (max-width: 900px) {
            .ds-lab-shell {
                grid-template-columns: 1fr;
                padding: 28px 24px;
            }
            .ds-avatar-stage {
                min-height: 210px;
            }
            .ds-avatar-halo {
                width: 204px;
                height: 204px;
            }
            .ds-avatar-ring {
                width: 230px;
                height: 78px;
            }
            .ds-avatar-head {
                width: 142px;
                height: 153px;
            }
            .ds-avatar-head::before {
                top: 31px;
                height: 75px;
            }
            .ds-avatar-eye { top: 65px; }
            .ds-avatar-eye.left { left: 41px; }
            .ds-avatar-eye.right { right: 41px; }
            .ds-avatar-mouth { top: 91px; }
            .ds-avatar-ear { top: 58px; }
        }
        </style>
        <section class="ds-lab-shell">
            <div class="ds-lab-content">
                <div class="ds-lab-kicker">DataSense AI · Model Workspace</div>
                <div class="ds-lab-title">Data Science <span>Lab</span></div>
                <div class="ds-lab-copy">
                    Move from an uploaded dataset to a verified model workflow.
                    DataSense AI prepares the fields, trains a baseline and
                    explains performance using results calculated on unseen
                    test data.
                </div>
                <div class="ds-lab-status">Predictive Modelling available now</div>
            </div>
            <div class="ds-avatar-stage" aria-label="DataSense AI avatar">
                <div class="ds-avatar-halo"></div>
                <div class="ds-avatar-ring"></div>
                <div class="ds-avatar-head">
                    <span class="ds-avatar-antenna"></span>
                    <span class="ds-avatar-ear left"></span>
                    <span class="ds-avatar-ear right"></span>
                    <span class="ds-avatar-eye left"></span>
                    <span class="ds-avatar-eye right"></span>
                    <span class="ds-avatar-mouth"></span>
                </div>
                <div class="ds-avatar-badge">DataSense AI</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_data_science_lab(
    df: pd.DataFrame | None,
    dataset_id: str,
    *,
    show_intro: bool = True,
) -> None:
    """Render capability cards and the first working modelling workflow."""
    if show_intro:
        render_data_science_lab_intro()

    st.markdown(
        """
        <div class="ds-capability-heading">Choose a data-science capability</div>
        <div class="ds-capability-sub">
            Start with predictive modelling. More model workflows will be
            activated here as they are built and tested.
        </div>
        """,
        unsafe_allow_html=True,
    )

    capability_key = f"ds_capability_{dataset_id}"
    old_value_map = {
        "📈 Predictive Modelling": "Predictive Modelling",
        "🔭 Forecasting": "Forecasting",
        "🧪 Hypothesis Testing": "Hypothesis Testing",
    }
    capability = old_value_map.get(
        st.session_state.get(capability_key),
        st.session_state.get(capability_key),
    )
    valid_capabilities = {
        "Predictive Modelling",
        "Forecasting",
        "Hypothesis Testing",
    }
    if capability not in valid_capabilities:
        capability = "Predictive Modelling"
    st.session_state[capability_key] = capability

    cards = [
        {
            "name": "Predictive Modelling",
            "class_name": "predictive",
            "icon": "📊",
            "status": "Available",
            "copy": "Predict a category or numeric outcome using verified test data.",
            "button": "OPEN MODEL",
        },
        {
            "name": "Forecasting",
            "class_name": "forecasting",
            "icon": "💡",
            "status": "Coming next",
            "copy": "Forecast future performance from validated dates and measures.",
            "button": "VIEW ROADMAP",
        },
        {
            "name": "Hypothesis Testing",
            "class_name": "hypothesis",
            "icon": "🧪",
            "status": "Planned",
            "copy": "Test whether differences between groups are statistically meaningful.",
            "button": "VIEW ROADMAP",
        },
    ]

    card_columns = st.columns(3, gap="medium")
    for column, card in zip(card_columns, cards):
        active_class = "is-active" if capability == card["name"] else ""
        with column:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="ds-model-card {card['class_name']} {active_class}">
                        <div class="ds-model-art">
                            <div class="ds-model-icon">{card['icon']}</div>
                        </div>
                        <div class="ds-model-status">{card['status']}</div>
                        <div class="ds-model-title">{card['name']}</div>
                        <div class="ds-model-copy">{card['copy']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    card["button"],
                    key=f"select_ds_card_{dataset_id}_{card['class_name']}",
                    use_container_width=True,
                    type="primary" if capability == card["name"] else "secondary",
                ):
                    st.session_state[capability_key] = card["name"]
                    st.rerun()

    if df is None:
        st.info(
            "Upload a CSV or Excel dataset above to open Predictive Modelling."
        )
        return

    if capability != "Predictive Modelling":
        capability_name = capability or "This capability"
        next_step = (
            "Forecast numeric performance over time using validated date and "
            "measure fields."
            if capability == "Forecasting"
            else "Compare groups and test whether an observed difference is "
            "statistically meaningful."
        )
        st.markdown(
            f"""
            <div class="ds-coming-soon">
                <strong>{capability_name} · Coming next</strong>
                <p>{next_step}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Planned after these options: Anomaly Detection and "
            "Clustering & Segmentation."
        )
        return

    if not SKLEARN_AVAILABLE:
        st.error(
            "Predictive Modelling requires scikit-learn. Install it with "
            "`pip install scikit-learn`, then restart DataSense AI."
        )
        st.caption(f"Import detail: {SKLEARN_IMPORT_ERROR}")
        return

    st.markdown("### 1. Choose the prediction target")
    target_column = st.selectbox(
        "What should DataSense AI predict?",
        options=df.columns.tolist(),
        key=f"predictive_target_{dataset_id}",
        help="The target is the outcome the model will learn to predict.",
    )

    inferred_type = infer_problem_type(df[target_column])
    type_choice = st.selectbox(
        "Problem type",
        options=[
            f"Auto-detect ({inferred_type})",
            "Classification",
            "Regression",
        ],
        key=f"predictive_type_{dataset_id}",
        help=(
            "Classification predicts categories such as cancelled/not "
            "cancelled. Regression predicts a numeric amount."
        ),
    )
    problem_type = (
        inferred_type if type_choice.startswith("Auto-detect") else type_choice
    )

    recommended, excluded = recommend_features(df, target_column)

    st.markdown("### 2. Choose the input features")
    feature_columns = st.multiselect(
        "Which columns may help predict the target?",
        options=[column for column in df.columns if column != target_column],
        default=recommended,
        key=f"predictive_features_{dataset_id}_{target_column}",
        help=(
            "DataSense AI excludes constants, dates, identifiers and "
            "high-cardinality text from the starter recommendation."
        ),
    )

    if excluded:
        with st.expander("Why were some columns not selected automatically?"):
            exclusion_table = pd.DataFrame(
                [
                    {"Column": column, "Reason": reason}
                    for column, reason in excluded.items()
                ]
            )
            st.dataframe(
                exclusion_table,
                use_container_width=True,
                hide_index=True,
            )

    test_percentage = st.slider(
        "Test-data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key=f"predictive_test_size_{dataset_id}",
        help=(
            "The model never sees the test rows while learning. They are "
            "used to measure performance on unseen data."
        ),
    )

    blocking, warnings, usable_rows = _readiness_checks(
        df,
        target_column,
        feature_columns,
        problem_type,
    )

    st.markdown("### 3. Review modelling readiness")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Problem", problem_type)
    metric_columns[1].metric("Usable Rows", f"{usable_rows:,}")
    metric_columns[2].metric("Features", len(feature_columns))
    metric_columns[3].metric(
        "Missing Target",
        f"{int(df[target_column].isna().sum()):,}",
    )

    if blocking:
        for issue in blocking:
            st.error(issue)
    else:
        st.success("The selected data passes the first modelling-readiness checks.")

    for warning in warnings:
        st.warning(warning)

    st.markdown("### 4. Train and evaluate a baseline")
    train_clicked = st.button(
        "Train baseline model",
        type="primary",
        disabled=bool(blocking),
        key=f"train_predictive_{dataset_id}",
    )

    configuration_id = _configuration_id(
        dataset_id,
        target_column,
        feature_columns,
        problem_type,
        test_percentage,
    )
    result_key = f"predictive_result_{dataset_id}"

    if train_clicked:
        try:
            with st.spinner("Training the baseline model..."):
                result = _train_baseline(
                    df,
                    target_column,
                    feature_columns,
                    problem_type,
                    test_percentage,
                )
            st.session_state[result_key] = {
                "configuration_id": configuration_id,
                "result": result,
            }
        except Exception as exc:
            st.error(f"The baseline model could not be trained: {exc}")

    stored = st.session_state.get(result_key)
    if not stored or stored.get("configuration_id") != configuration_id:
        return

    result = stored["result"]
    st.markdown(f"#### Test results · {result['model_name']}")
    st.caption(
        f"Trained on {result['training_rows']:,} rows and evaluated on "
        f"{result['test_rows']:,} unseen rows."
    )
    if result["split_note"]:
        st.warning(result["split_note"])

    metrics = result["metrics"]
    metric_boxes = st.columns(len(metrics))
    for box, (label, value) in zip(metric_boxes, metrics.items()):
        if result["problem_type"] == "Classification":
            box.metric(label, f"{value:.3f}")
        else:
            box.metric(label, f"{value:,.3f}")

    _render_verified_interpretation(result)

    if result.get("confusion_matrix") is not None:
        st.markdown("#### Confusion matrix")
        st.dataframe(
            result["confusion_matrix"],
            use_container_width=True,
        )

    st.markdown("#### Sample test predictions")
    st.dataframe(
        result["predictions"].head(50),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download test predictions",
        data=result["predictions"].to_csv(index=False).encode("utf-8"),
        file_name=f"{target_column}_baseline_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"download_predictions_{configuration_id}",
    )

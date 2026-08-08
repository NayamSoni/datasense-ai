# DataSense AI

A local-first AI data analyst built with Streamlit, Pandas, Altair, Plotly, and Ollama.

DataSense AI allows users to upload CSV or Excel datasets, assess data quality, perform calculation-backed analysis, create visualisations, and receive evidence-linked business insights through natural-language questions.

> **Project status:** Version 2 runs locally using Ollama and `llama3.2:3b`. A cloud-hosted demo is not currently available.

## Overview

Business users often spend significant time checking data quality, creating calculations, selecting suitable charts, and translating analytical results into useful recommendations.

DataSense AI brings these activities into one guided workflow:

1. Upload a CSV or Excel file.
2. Review the dataset structure and health score.
3. Investigate missing values, duplicates, outliers, and other quality issues.
4. Apply user-selected cleaning actions.
5. Ask analytical questions in natural language.
6. Generate calculations, pivots, charts, insights, and recommendations.

The language model helps interpret questions and explain results, while Pandas performs the underlying calculations.

## Key Features

### Data quality and cleaning

* Detects missing values and affected rows
* Identifies duplicate records
* Detects potential outliers using the IQR method
* Flags likely incorrect data types
* Identifies constant columns
* Highlights high-cardinality columns
* Provides user-controlled cleaning actions
* Maintains a cleaning audit log
* Supports cleaned-data CSV export

### Dataset Health Score

Generates an overall diagnostic score using five measurable components:

| Component           | Weight |
| ------------------- | -----: |
| Completeness        |    25% |
| Consistency         |    20% |
| Duplicate quality   |    20% |
| Missing-row quality |    20% |
| Outlier quality     |    15% |

```text
Overall Health Score =
25% Completeness
+ 20% Consistency
+ 20% Duplicate Quality
+ 20% Missing-Row Quality
+ 15% Outlier Quality
```

The score is diagnostic rather than prescriptive. Outliers, constant columns, and high-cardinality fields are not automatically treated as errors. The user decides whether they should be cleaned.

### Calculation-backed analytics

* Top and bottom performer analysis
* Month-over-month analysis
* Year-over-year analysis
* Pivot-table generation
* Correlation analysis
* Distribution analysis
* Outlier analysis
* Pareto analysis
* Drag-and-drop visualisation
* Predictive modelling workspace

### Conversational analysis

DataSense AI retains the context of follow-up questions.

For example:

```text
Show sales by region
→ Only for 2023
→ Now only California
```

The application carries the relevant metric, grouping, and filters into the subsequent analysis.

### Business insights

* Generates insights from computed results
* Links observations to supporting evidence
* Identifies performance patterns and potential operational gaps
* Presents recommendations as hypotheses to validate
* Avoids presenting correlation as proven causation

## Grounding Principle

The LLM is used to interpret questions and produce general explanations. Pandas performs the actual calculations.

Automatic insights and recommendations are generated only from computed evidence. Recommendations are presented as hypotheses requiring business validation rather than as confirmed causal conclusions.

## Technology Stack

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Core application development        |
| Streamlit    | Interactive user interface          |
| Pandas       | Data transformation and calculation |
| NumPy        | Numerical operations                |
| Altair       | Declarative visualisation           |
| Plotly       | Interactive charts                  |
| Scikit-learn | Predictive modelling                |
| Ollama       | Local LLM execution                 |
| Llama 3.2 3B | Question planning and explanations  |
| OpenPyXL     | Excel-file support                  |

## Project Structure

```text
datasense-ai/
├── app.py
├── conversation_memory.py
├── data_quality.py
├── feedback_memory.py
├── insights_engine.py
├── intent_agent.py
├── llm_agent.py
├── pandas_agent.py
├── predictive_modeling.py
├── prompts.py
├── query_planner.py
├── schema.py
├── theme.py
├── utils.py
├── visualization.py
├── styles.css
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   └── robot.png
└── tests/
```

The robot image is optional. If `assets/robot.png` is unavailable, the application falls back to a robot emoji.

## Running the Application Locally

### 1. Install and prepare Ollama

Install Ollama, start it locally, and download the required model:

```bash
ollama pull llama3.2:3b
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Current Limitations

* Ollama must be installed and running on the user’s computer.
* Response quality can vary because the application uses a lightweight local model.
* Processing very large datasets may be limited by the computer’s available memory.
* Generated recommendations require business review and domain validation.
* Data-cleaning decisions remain under the user’s control.

## Privacy

DataSense AI is designed as a local-first application. Dataset calculations and LLM inference run on the user’s machine through Pandas and Ollama rather than relying on a paid external LLM API.

## Future Improvements

* Add a shareable hosted demonstration
* Expand automated testing
* Improve predictive-model evaluation and explainability
* Add support for additional data sources
* Strengthen analysis validation and error handling

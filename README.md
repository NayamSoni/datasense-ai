# DataSense AI — Version 2

> An agentic data analyst that turns structured data into quality checks, calculations, visualisations, business insights, and predictive models.

DataSense AI is a Streamlit application built to behave less like a generic chatbot and more like a practical data analyst. It combines deterministic Python/Pandas calculations with a local Ollama model for interpretation and conversational guidance.

Users can analyse data from either:

- CSV or Excel uploads
- A live Supabase PostgreSQL demo table

When database rows change, DataSense AI can refresh the table and rerun the same analysis pipeline without another file upload.

## What it demonstrates

- Modular analytics application design
- Natural-language intent detection and query planning
- Calculation-backed answers instead of invented figures
- Data quality profiling and user-controlled cleaning
- Cloud PostgreSQL connectivity from Streamlit
- Stateful follow-up analysis and learned corrections
- Classification and regression workflows
- Local LLM integration without a paid AI API

## Current features

### Multiple data sources

- Upload CSV and XLSX files
- Connect to `public.demo_sales` in Supabase PostgreSQL
- Refresh database data without restarting the application
- Detect changed database content and clear stale analysis results

### Data quality

- Missing-value analysis
- Duplicate detection
- Type and consistency checks
- IQR-based outlier detection
- Constant and high-cardinality column checks
- Dataset Health Score
- Cleaning suggestions with an audit log

### Business analysis

- KPI and statistical summaries
- Aggregations, rankings, Top/Bottom analysis, and Pareto analysis
- Contribution percentages calculated against the full breakdown
- MoM and YoY analysis
- Pivot, correlation, distribution, and outlier analysis
- Calculation-grounded insights and recommendations
- Editable decision reports with supporting charts

### Conversational workspace

- Intent detection and structured query planning
- Follow-up question memory
- Deterministic Pandas execution
- Local Ollama reasoning
- Experimental learned correction rules

### Visualisation

- Normal and 3D charts
- Drag-and-drop visualisation shelves
- User-controlled dimensions, measures, aggregations, colours, and Top N

### Data Science Lab

- Target selection
- Automatic classification/regression detection
- Dataset readiness checks
- Train/test workflow
- Model evaluation and understandable predictions

## Architecture

```mermaid
flowchart TD
    A[CSV / Excel] --> C[Pandas DataFrame]
    B[Supabase PostgreSQL] --> C
    C --> D[Profile and Quality Checks]
    D --> E[Intent and Query Planning]
    E --> F[Deterministic Pandas Execution]
    F --> G[Insights, Charts, Reports]
    F --> H[Data Science Lab]
    G --> I[Local Ollama Interpretation]
```

The LLM helps interpret requests and explain results. Numerical calculations remain in Python/Pandas.

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Altair and Plotly
- scikit-learn
- Ollama with `llama3.2:3b`
- PostgreSQL and Supabase
- SQLAlchemy and Psycopg
- pytest

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/NayamSoni/datasense-ai.git
cd datasense-ai
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the local AI model

Install [Ollama](https://ollama.com/) and run:

```bash
ollama pull llama3.2:3b
```

### 5. Start DataSense AI

```bash
streamlit run app.py
```

CSV/Excel analysis works without configuring Supabase.

## Optional Supabase database setup

DataSense AI expects the demo database table at `public.demo_sales`.

### 1. Create a free Supabase project

Create a project at [database.new](https://database.new/) and save its database password securely.

### 2. Create the sample table

Run this once in the Supabase SQL Editor:

```sql
CREATE TABLE public.demo_sales (
    order_id BIGINT PRIMARY KEY,
    order_date DATE NOT NULL,
    city TEXT NOT NULL,
    product_category TEXT NOT NULL,
    customer_segment TEXT NOT NULL,
    units INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    discount_pct NUMERIC(5, 2) NOT NULL,
    revenue NUMERIC(12, 2) NOT NULL,
    profit NUMERIC(12, 2) NOT NULL,
    returned BOOLEAN NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.demo_sales ENABLE ROW LEVEL SECURITY;

INSERT INTO public.demo_sales (
    order_id, order_date, city, product_category, customer_segment,
    units, unit_price, discount_pct, revenue, profit, returned
)
VALUES
    (1, CURRENT_DATE, 'Bengaluru', 'Electronics', 'Consumer', 12, 1499, 10, 16189.20, 2428.38, FALSE),
    (2, CURRENT_DATE, 'Mumbai', 'Furniture', 'Corporate', 8, 2499, 5, 18992.40, 3038.78, FALSE),
    (3, CURRENT_DATE, 'Delhi', 'Office Supplies', 'Small Business', 20, 499, 0, 9980.00, 1497.00, FALSE),
    (4, CURRENT_DATE, 'Pune', 'Appliances', 'Consumer', 7, 3499, 15, 20819.05, 3331.05, TRUE),
    (5, CURRENT_DATE, 'Hyderabad', 'Accessories', 'Corporate', 25, 899, 5, 21351.25, 3202.69, FALSE);
```

### 3. Add the database connection locally

Copy the Supabase **Session pooler** URI. Create `.streamlit/secrets.toml`:

```toml
[connections.datasense_db]
url = "postgresql+psycopg://USERNAME:PASSWORD@HOST:5432/postgres?sslmode=require"
```

Replace every placeholder with your own connection details.

Do not commit this file. The repository's `.gitignore` excludes it.

### 4. Test the connection

```bash
python -c 'import tomllib; from sqlalchemy import create_engine, text; config = tomllib.load(open(".streamlit/secrets.toml", "rb")); engine = create_engine(config["connections"]["datasense_db"]["url"]); connection = engine.connect(); print("Connected rows:", connection.execute(text("SELECT COUNT(*) FROM public.demo_sales")).scalar()); connection.close()'
```

Then start the app, select **Demo database**, and click **Refresh data**.

## Security and privacy

- No paid AI API key is required.
- Ollama model inference runs locally.
- Uploaded files are processed in application memory.
- The optional Supabase dataset is cloud-hosted and fetched into the application.
- Database credentials stay in `.streamlit/secrets.toml` and must never be committed.
- The current demo connector executes a fixed read query.
- Use a dedicated read-only PostgreSQL role before deploying the connector publicly.

## Tests

Run:

```bash
python -m pytest -q
```

Current automated checks cover core analytical behaviour, including data quality, cleaning, grounded insights, and follow-up memory.

## Project status

DataSense AI V2 is an active portfolio project. The database connector, quality workflow, analytical engine, visualisation workspace, and first predictive-modelling workflow are functional.

Planned improvements:

- Dataset blending and multi-table analysis
- Time-series forecasting
- Hypothesis testing
- Anomaly detection
- Additional read-only database connectors
- Expanded automated test coverage

## License

See the repository's `LICENSE` file.

---

Built as a practical learning project in analytics, data science, and agentic AI.

**Still learning. Still building.**

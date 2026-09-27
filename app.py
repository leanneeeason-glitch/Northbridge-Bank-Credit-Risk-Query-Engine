"""
Northbridge Bank - Commercial Lending Credit Risk Query Engine
================================================================
Streamlit app version of the notebook-based natural-language query engine.

Pipeline (unchanged from the notebook):
    1. classify_intent   -> route to a Verified Query template or fresh SQL generation
    2. generate_query    -> LLM writes SQL for novel questions
    3. validate_query    -> 5-check validation gate (read-only, schema, EXPLAIN, LLM relevance, template integrity)
    4. retry_generation  -> one retry on the generated track if validation fails
    5. execute_query     -> runs the gate-passed SQL, flags result "reasonableness"
    6. generate_response -> LLM turns the result set into a business-facing narrative

Required files in the same directory as this app.py:
    - credit_risk_portfolio.db   (SQLite database)
    - test_queries.csv           (optional - only needed for the "Run test suite" sidebar feature)

Required configuration (see bottom of file / README notes for details):
    - OPENAI_API_KEY   (required)
    - OPENAI_API_BASE  (optional - only needed if you use a custom/proxy OpenAI endpoint)
"""

import json
import os
import re
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

DB_PATH = "credit_risk_portfolio.db"
TEST_QUERIES_PATH = "test_queries.csv"

# ----------------------------------------------------------------------------
# Credential loading
# Priority: Streamlit secrets  ->  environment variables  ->  local config.json
# (config.json is convenient for local runs; use st.secrets/env vars when deployed)
# ----------------------------------------------------------------------------
def load_credentials():
    api_key = None
    api_base = None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE")

    if not api_key and Path("config.json").exists():
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    return api_key


API_KEY_FOUND = load_credentials()

# ----------------------------------------------------------------------------
# Cached resources: LLMs and read-only DB connection
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llms():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)        # primary reasoning, classification, generation
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)   # validation and evaluation
    return llm, evaluator_llm


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    # Read-only connection, exactly as in the notebook
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


@st.cache_data(show_spinner=False)
def get_database_schema():
    return """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ----------------------------------------------------------------------------
# Verified Query Template Library (10 pre-approved SQL templates)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_verified_query_library():
    sql_1 = """
SELECT sector_name,
    ROUND(SUM(total_outstanding)/1000000.0, 1) AS outstanding_exposure,
    ROUND(SUM(CASE WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN total_outstanding ELSE 0 END)/1000000.0, 2) as NPA_exposure
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sector_name
ORDER BY outstanding_exposure DESC
"""

    sql_2 = """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding)/1000000.0, 2) AS outstanding_exposure,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY outstanding_exposure DESC
"""

    sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(DISTINCT loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount)/1000000.0, 2) AS ead_amount,
    ROUND(SUM(ecl_amount)/1000000.0, 2) AS ecl_amount
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

    sql_4 = """
SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning p
    JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
    JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_coverage_ratio DESC
"""

    sql_5 = """
SELECT
    borrower_name,
    sector_code,
    ROUND(total_outstanding/1000000.0, 2) AS outstanding_exposure,
    asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10
"""

    sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding)/1000000.0, 2) AS outstanding_exposure
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY outstanding_exposure DESC
LIMIT 5
"""

    sql_7 = """
SELECT
    loan_account_number,
    borrower_name,
    sector_code,
    ROUND(total_outstanding/1000000.0, 2) AS outstanding_exposure,
    days_past_due,
    asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

    sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding)/1000000.0, 2) AS outstanding_exposure
FROM loan_master
GROUP BY dpd_bucket
ORDER BY dpd_bucket ASC
"""

    sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

    sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount)/1000000.0, 2) AS total_ecl
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC
"""

    return {
        "VQ1": {
            "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
            "sql": sql_1,
        },
        "VQ2": {
            "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
            "sql": sql_2,
        },
        "VQ3": {
            "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
            "sql": sql_3,
        },
        "VQ4": {
            "description": "Average provision coverage ratio by sector for the latest reporting quarter",
            "sql": sql_4,
        },
        "VQ5": {
            "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
            "sql": sql_5,
        },
        "VQ6": {
            "description": "Top 5 largest exposures aggregated at the business group level",
            "sql": sql_6,
        },
        "VQ7": {
            "description": "All overdue loan accounts with their days past due and asset classification",
            "sql": sql_7,
        },
        "VQ8": {
            "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
            "sql": sql_8,
        },
        "VQ9": {
            "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
            "sql": sql_9,
        },
        "VQ10": {
            "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
            "sql": sql_10,
        },
    }


# ----------------------------------------------------------------------------
# Pipeline tools (logic preserved exactly from the notebook)
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library, llm):
    """
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    """
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a credit risk analyzer for a commercial banking analytics system. Your job is to decide whether a business user's questions can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
USER Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'exposure' refers to 'total_outstanding', 'sector' refers to 'sector_code', and 'sector_name', and DPD refers to 'days_past_due'.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'Show me the sectors') should NOT match a template that returns an aggregate count.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    """
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    """
    generation_prompt = f"""

### ROLE
You are a senior SQLite developer specializing in commercial banking analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using sector_name or sector_code where relevant (for example, 'Real Estate' maps to SEC_RE, 'Infrastructure' maps to SEC_INFRA).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing.

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.

"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    """
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    """
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or "Name" in str(t.ttype)]
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about commercial banking operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    """
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    """
    retry_prompt = f"""

### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    """
    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    """
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    """
    response_prompt = f"""

### ROLE
You are a credit risk analyst writing a concise business response for a routine lending portfolio question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific sector, product type, loan category, ifrs9 stage, or provisioning date, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not compute percentages, ratios, or totals yourself. If a percentage is not present in the table, do not state one.
5. Flag anything notable, such as a total outstanding loan amount significantly different than peers or days past due greater than 60 days.
6. Use clear, professional language suitable for a lending portfolio memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. Loan amounts and outstanding exposures are already converted to millions of USD. State the units for these amounts a "M" and do not convert to millions.
9. State the unit for every number, inferred from its column name: _amount as "$XM", _count as plain count, _month as "X months", days_ as "X days". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm):
    """
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    """
    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log["candidate_sql"] = candidate_sql

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log["query_id"])
    log["gate_result"] = gate

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log["gate_result"] = gate

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        return {"log": log, "dataframe": None, "warnings": [], **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], llm, log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")

    return {"log": log, "dataframe": df, "warnings": exec_result["warnings"], **log}


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def render_trace(result: dict):
    """Render the pipeline trace (route, SQL, confidence, warnings) for one answer."""
    log = result["log"]
    cols = st.columns(4)
    cols[0].metric("Route", log["route"] or "-")
    cols[1].metric("Template", log["query_id"] or "—")
    conf = log["confidence"]
    cols[2].metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
    cols[3].metric("Rows Returned", log["row_count"] if log["row_count"] is not None else "—")

    if log.get("match_reason"):
        st.caption(f"Routing reason: {log['match_reason']}")
    if log.get("retry_used"):
        st.warning("A validation retry was required before this query was accepted.")

    if log["escalated"]:
        st.error(log["narrative"])
        with st.expander("Failed SQL and validation details"):
            st.code(log["candidate_sql"], language="sql")
            st.json(log["gate_result"])
        return

    st.markdown(log["narrative"])

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    with st.expander("Executed SQL and full result set"):
        st.code(log["executed_sql"], language="sql")
        st.dataframe(result["dataframe"], use_container_width=True)


def main():
    st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
    st.caption(
        "Ask routine commercial lending portfolio questions in plain English. "
        "Verified questions are answered from a pre-approved template library; "
        "novel questions are answered with freshly generated, validated SQL."
    )

    if not API_KEY_FOUND:
        st.error(
            "No OpenAI API key found. Add `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`) "
            "to Streamlit secrets, environment variables, or a local `config.json` file, then reload the app."
        )
        st.stop()

    if not Path(DB_PATH).exists():
        st.error(f"Database file `{DB_PATH}` was not found next to app.py. Please add it and reload.")
        st.stop()

    llm, evaluator_llm = get_llms()
    conn = get_connection(DB_PATH)
    schema_context = get_database_schema()
    query_library = get_verified_query_library()

    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, result) tuples

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Verified Query Library")
        st.caption("10 pre-approved templates covering the most common recurring questions.")
        for qid, entry in query_library.items():
            with st.expander(f"{qid}"):
                st.write(entry["description"])
                st.code(entry["sql"].strip(), language="sql")

        st.divider()
        st.header("Example questions")
        examples = [
            "How many loans are not past due?",
            "What is the sector-wise NPA exposure?",
            "Show me the DPD aging profile of the portfolio.",
            "Which loans have been restructured?",
            "What is the average interest rate by sector?",
            "How has Stage 3 exposure changed across quarters?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state["pending_question"] = ex

        st.divider()
        st.header("Test suite")
        st.caption("Runs all rows in test_queries.csv through the pipeline and compares to the expected route/template.")
        run_tests = st.button("Run test suite", use_container_width=True)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # ---------------- Test suite ----------------
    if run_tests:
        if not Path(TEST_QUERIES_PATH).exists():
            st.error(f"`{TEST_QUERIES_PATH}` was not found next to app.py.")
        else:
            ground_truth = pd.read_csv(TEST_QUERIES_PATH)
            evaluation_rows = []
            with st.spinner(f"Running {len(ground_truth)} test cases through the pipeline..."):
                for _, gt in ground_truth.iterrows():
                    tr = run_pipeline(gt["User Query"], conn, query_library, schema_context, llm, evaluator_llm)
                    evaluation_rows.append(
                        {
                            "Test Case": gt["Test Case"],
                            "User Query": gt["User Query"],
                            "Expected Route": gt["Expected Route"],
                            "Actual Route": tr["route"],
                            "Route Match": tr["route"] == gt["Expected Route"],
                            "Expected Query ID": gt["Expected Query ID"],
                            "Actual Query ID": tr["query_id"],
                            "Query ID Match": (
                                pd.isna(gt["Expected Query ID"]) and pd.isna(tr["query_id"])
                            )
                            or tr["query_id"] == gt["Expected Query ID"],
                            "Confidence": tr["confidence"],
                            "Rows Returned": tr["row_count"],
                        }
                    )
            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df["Route Match"].mean() * 100
            verified_mask = evaluation_df["Expected Route"].str.strip().str.lower() == "verified"
            query_accuracy = (
                evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100 if verified_mask.any() else float("nan")
            )
            numeric_confidence = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
            average_confidence = numeric_confidence.mean()

            st.subheader("Evaluation against ground truth")
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%" if pd.notna(query_accuracy) else "—")
            m3.metric("Average Confidence Score", f"{average_confidence:.2f}" if pd.notna(average_confidence) else "—")
            st.dataframe(evaluation_df, use_container_width=True)

    # ---------------- Conversation history ----------------
    for question, result in st.session_state.history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_trace(result)

    # ---------------- New question ----------------
    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about the lending portfolio...")
    question = question or pending

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Classifying intent, building and validating SQL..."):
                try:
                    result = run_pipeline(question, conn, query_library, schema_context, llm, evaluator_llm)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    result = None
            if result is not None:
                render_trace(result)
                st.session_state.history.append((question, result))


if __name__ == "__main__":
    main()

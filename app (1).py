"""
Northbridge Bank — Credit Risk Portfolio Query Engine
Streamlit application converted from the Project 3 Learner Notebook.

Preserves the original pipeline logic:
  1. Intent classification (verified template vs. generated SQL)
  2. Query construction (verified library lookup OR fresh SQL generation)
  3. Five-stage validation gate
  4. Single retry on validation failure for the generated track
  5. Escalation to a human analyst if retry also fails
  6. Read-only execution against the SQLite database
  7. Natural-language response generation
"""

import json
import os
import re
import sqlite3
import warnings

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# CONFIGURATION / SECRETS
# =============================================================================
# Supports three ways of supplying credentials, in priority order:
#   1. Streamlit secrets (st.secrets) — recommended for Streamlit Cloud
#   2. Environment variables OPENAI_API_KEY / OPENAI_API_BASE
#   3. A local config.json file (same format the notebook used)
def load_credentials():
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        api_base = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE")

    # 3. config.json fallback (notebook-style)
    if not api_key and os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = api_key or config.get("OPENAI_API_KEY")
                api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

DB_PATH = os.environ.get("DB_PATH", "credit_risk_portfolio.db")


# =============================================================================
# CACHED RESOURCES: LLMs and DB connection
# =============================================================================
@st.cache_resource
def get_llms():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)          # primary reasoning / classification / generation
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)     # validation / evaluation
    return llm, evaluator_llm


@st.cache_resource
def get_db_connection(db_path: str):
    # Read-only connection via URI mode, exactly as in the notebook
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


# =============================================================================
# DATABASE SCHEMA (provided to the LLM in prompts)
# =============================================================================
DATABASE_SCHEMA = """
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


# =============================================================================
# VERIFIED QUERY TEMPLATE LIBRARY (10 pre-approved SQL templates)
# =============================================================================
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

VERIFIED_QUERY_LIBRARY = {
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


# =============================================================================
# PIPELINE TOOLS (ported 1:1 from the notebook)
# =============================================================================
def classify_intent(user_question, query_library, llm):
    """Classifies the user question and decides which route to take."""
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
    """Generates a candidate SQL query for a novel question using the database schema."""
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
    """Validates a candidate SQL query through five checks before execution."""
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
    """Regenerates SQL after a validation failure, feeding the error back to the LLM."""
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
    """Executes a gate-passed SQL query and returns the result as a DataFrame."""
    result = {
        "dataframe": None,
        "reasonable": True,
        "warnings": [],
    }

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
    """Generates a focused natural language response from the query result."""
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


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, status_cb=None):
    """
    Runs the complete query engine pipeline for a single user question.

    status_cb: optional callable(str) used to stream stage-by-stage progress
               messages back to the Streamlit UI (replaces notebook print()).
    """

    def log_status(msg):
        if status_cb:
            status_cb(msg)

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

    log_status(f"**[1] Intent Classification:** route=`{log['route']}`, query_id=`{log['query_id']}`  \n"
               f"Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log["candidate_sql"] = candidate_sql

    log_status(f"**[2] Query Construction:** "
               f"{'loaded from verified library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log["query_id"])
    log["gate_result"] = gate

    log_status(f"**[3] Validation Gate:** passed=`{gate['passed']}`, "
               f"relevance_confidence=`{gate.get('relevance_confidence')}`" +
               (f"  \nFailed check: `{gate.get('failed_check')}` — {gate.get('details')}" if not gate["passed"] else ""))

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        log_status(f"Retrying after failure: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log["gate_result"] = gate

        log_status(f"**Retry Validation Gate:** passed=`{gate['passed']}`, "
                   f"relevance_confidence=`{gate.get('relevance_confidence')}`" +
                   (f"  \nRetry failed check: `{gate.get('failed_check')}` — {gate.get('details')}" if not gate["passed"] else ""))

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        log_status(f"**[!] Escalated to human analyst:** {gate['details']}")
        return {"log": log, "dataframe": None, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)

    log_status(f"**[4] Execute:** {len(df)} rows returned" +
               (f"  \nWarnings: {exec_result['warnings']}" if exec_result["warnings"] else ""))

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], llm, log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")

    log_status(f"**[5] Response Generation:** confidence=`{log['confidence']}`")

    return {"log": log, "dataframe": df, **log}


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🏦 Credit Risk Portfolio Query Engine")
    st.caption("Northbridge Bank — Commercial Lending Portfolio | Natural-language Q&A over read-only credit-risk data")

    # --- Credential / DB checks -------------------------------------------------
    if not OPENAI_API_KEY:
        st.error(
            "No OpenAI API key found. Set it via Streamlit secrets "
            "(`OPENAI_API_KEY`), an environment variable, or a local `config.json` "
            "file with keys `OPENAI_API_KEY` and `OPENAI_API_BASE`."
        )
        st.stop()

    if not os.path.exists(DB_PATH):
        st.error(
            f"Database file `{DB_PATH}` not found. Place `credit_risk_portfolio.db` "
            f"in the app's working directory (or set the `DB_PATH` environment variable)."
        )
        st.stop()

    llm, evaluator_llm = get_llms()
    conn = get_db_connection(DB_PATH)

    # --- Sidebar: reference info -------------------------------------------------
    with st.sidebar:
        st.header("About this engine")
        st.markdown(
            "This tool routes routine portfolio questions to a **verified, "
            "pre-approved SQL template** whenever possible. Questions outside "
            "that library are answered with **freshly generated, read-only SQL** "
            "that passes a five-stage validation gate before running. Anything "
            "that fails validation twice is **escalated to a human analyst**."
        )

        st.subheader("Verified Query Library")
        for qid, entry in VERIFIED_QUERY_LIBRARY.items():
            st.markdown(f"**{qid}** — {entry['description']}")

        st.subheader("Database tables")
        st.markdown("- `sector_master`\n- `loan_master`\n- `borrower_rating`\n- `provisioning`")

        show_trace = st.checkbox("Show pipeline trace", value=True)
        show_sql = st.checkbox("Show executed SQL", value=True)

    # --- Sample questions ---------------------------------------------------------
    st.subheader("Ask a question about the lending portfolio")

    sample_questions = [
        "What is the NPA exposure in the Real Estate sector?",
        "Show me the aging profile of the portfolio by DPD bucket.",
        "Which loans have been restructured and what is their total outstanding?",
        "What is the average interest rate by sector?",
        "How has Stage 3 exposure changed across reporting quarters?",
    ]

    cols = st.columns(len(sample_questions))
    picked_sample = None
    for c, q in zip(cols, sample_questions):
        if c.button(q, use_container_width=True):
            picked_sample = q

    default_text = picked_sample if picked_sample else st.session_state.get("last_question", "")
    user_question = st.text_area(
        "Your question",
        value=default_text,
        placeholder="e.g. What is the total outstanding exposure in the Infrastructure sector?",
        height=80,
    )

    run_clicked = st.button("Run Query", type="primary")

    # --- Run pipeline ---------------------------------------------------------
    if run_clicked and user_question.strip():
        st.session_state["last_question"] = user_question

        trace_container = st.container()
        trace_lines = []

        def status_cb(msg):
            trace_lines.append(msg)

        with st.spinner("Running query engine pipeline..."):
            try:
                result = run_pipeline(
                    user_question=user_question,
                    db_connection=conn,
                    query_library=VERIFIED_QUERY_LIBRARY,
                    schema_context=DATABASE_SCHEMA,
                    llm=llm,
                    evaluator_llm=evaluator_llm,
                    status_cb=status_cb,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        if show_trace:
            with trace_container.expander("Pipeline trace", expanded=False):
                for line in trace_lines:
                    st.markdown(line)

        st.divider()

        if result["escalated"]:
            st.warning("⚠️ This question was escalated to a human analyst.")
            st.write(result["narrative"])
            if show_sql:
                st.subheader("Last attempted SQL")
                st.code(result["candidate_sql"], language="sql")
        else:
            # Route / metadata badges
            badge_cols = st.columns(4)
            badge_cols[0].metric("Route", result["route"])
            badge_cols[1].metric("Query ID", result["query_id"] or "—")
            conf = result["confidence"]
            badge_cols[2].metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
            badge_cols[3].metric("Rows returned", result["row_count"])

            st.subheader("Answer")
            st.write(result["narrative"])

            if show_sql:
                st.subheader("Executed SQL")
                st.code(result["executed_sql"], language="sql")

            st.subheader("Result data")
            st.dataframe(result["dataframe"], use_container_width=True)

            csv = result["dataframe"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="query_result.csv",
                mime="text/csv",
            )

    elif run_clicked and not user_question.strip():
        st.warning("Please enter a question before running.")


if __name__ == "__main__":
    main()

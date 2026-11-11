# =============================================================================
# Northbridge Bank | Credit Risk Query Engine (Streamlit version)
#
# Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb
#
# Pipeline (unchanged from the notebook):
#   classify_intent -> (verified template | generate_query) -> validate_query
#   -> retry_generation (once, generated route only) -> escalate if still failing
#   -> execute_query -> generate_response
#
# Changes made for Streamlit (logic, SQL and prompts are otherwise preserved):
#   - config.json / Google Drive mount replaced by env vars, st.secrets or an
#     optional local config.json, plus a DB_PATH setting
#   - print() trace output is collected into a trace list and shown in the UI
#   - SQLite connection opened read-only with check_same_thread=False
#   - Added: in-session audit log (optional JSONL file), result warnings,
#     timestamps, elapsed time
# =============================================================================

import json
import os
import re
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

APP_DIR = Path(__file__).resolve().parent


def get_setting(name, default=None):
    """Read a setting from environment variables first, then st.secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def load_config_json(file_name='config.json'):
    """Optional fallback: the same config.json format used in the notebook."""
    path = APP_DIR / file_name
    if path.exists():
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def configure_credentials():
    """
    Mirrors the notebook: OPENAI_API_KEY and OPENAI_API_BASE are stored as
    environment variables (the notebook sets OPENAI_BASE_URL; OPENAI_API_BASE is
    also set because some langchain-openai versions read that name instead).
    Returns (api_key_found, api_base_found).
    """
    config = load_config_json()
    api_key = get_setting("OPENAI_API_KEY") or config.get("OPENAI_API_KEY")
    api_base = get_setting("OPENAI_API_BASE") or config.get("OPENAI_API_BASE")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base
    return bool(api_key), bool(api_base)


def resolve_db_path():
    raw = get_setting("DB_PATH", "credit_risk_portfolio.db")
    path = Path(raw)
    if not path.is_absolute():
        path = APP_DIR / path
    return path


@st.cache_resource(show_spinner=False)
def get_connection(db_path_str):
    """
    Read-only connection (URI mode with ?mode=ro) so no write operation is possible.
    check_same_thread=False is needed because Streamlit runs scripts in worker threads.
    """
    uri = f"{Path(db_path_str).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_resource(show_spinner=False)
def init_llms():
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    evaluator_llm = ChatOpenAI(model='gpt-4o', temperature=0)
    return llm, evaluator_llm


# =============================================================================
# DATABASE SCHEMA PROVIDED TO THE LLM (verbatim from the notebook)
# =============================================================================

database_schema = """
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
# GROUND-TRUTH TEST CASES (built programmatically, as in the notebook)
# =============================================================================

credit_risk_data = {
    'test_id': ['TQ001', 'TQ002', 'TQ003', 'TQ004', 'TQ005'],
    'question': [
        'Provide a sector-wise breakdown of our total outstanding loan exposure and NPA amounts.',
        'What is the total portfolio outstanding and loan count across different loan categories?',
        'Show me the top 10 largest individual loan exposures at the borrower level.',
        'Which borrowers were downgraded in the latest rating cycle, and what are their current ratings and PD estimates?',
        'What is the physical home address and personal email of our top relationship managers?'
    ],
    'expected_route': ['verified', 'verified', 'verified', 'generated', 'escalate'],
    'expected_query_id': ['VQ1', 'VQ2', 'VQ5', None, None],
    'expected_answer': [
        'Sector-wise breakdown of outstanding and NPA exposure.',
        'Outstanding amount and loan count grouped by Corporate, Mid-Corporate, and SME.',
        'List of top 10 individual borrower loans sorted by total outstanding principal and interest.',
        'List of downgraded borrowers on 2025-09-30.',
        'Escalated: Personal identification information is restricted and not present in the analytical database.'
    ]
}

ground_truth = pd.DataFrame(credit_risk_data)

# =============================================================================
# VERIFIED QUERY TEMPLATE LIBRARY (SQL verbatim from the notebook)
# =============================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_million
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_million DESC;
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
    COUNT(loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_million DESC;
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_million,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS average_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY average_provision_coverage_ratio DESC;
"""

# VQ5: Top 10 Loan Exposures (the LIMIT 10 is appended in the library below, as in the notebook)
sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != ''
GROUP BY group_name
ORDER BY total_outstanding_million DESC
LIMIT 5;
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE
        WHEN dpd_bucket = '0 (Current)' THEN 1
        WHEN dpd_bucket = '1-30' THEN 2
        WHEN dpd_bucket = '31-60' THEN 3
        WHEN dpd_bucket = '61-90' THEN 4
        ELSE 5
    END;
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating AS current_internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1.strip().rstrip(';').strip()
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2.strip().rstrip(';').strip()
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3.strip().rstrip(';').strip()
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4.strip().rstrip(';').strip()
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5.strip().rstrip(';').strip() + " LIMIT 10"
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6.strip().rstrip(';').strip()
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7.strip().rstrip(';').strip()
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8.strip().rstrip(';').strip()
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9.strip().rstrip(';').strip()
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10.strip().rstrip(';').strip()
    }
}

# =============================================================================
# TOOL DEFINITIONS
# `llm` and `evaluator_llm` are module-level globals assigned in the runtime
# initialisation section further down (same names as in the notebook).
# =============================================================================


def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified, generated, or escalate),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a precise routing agent for Northbridge Bank's Credit Risk Analytics Query Engine.
Your task is to classify incoming user questions and route them to \"verified\", \"generated\", or \"escalate\".

### VERIFIED TEMPLATE LIBRARY
{library_descriptions}

### ROUTING RULES
1. ROUTE TO VERIFIED: If the user's question asks for any of the routine reports matching the library descriptions above, select \"route\": \"verified\" and provide the exact \"query_id\".
   - Note: Questions asking for total portfolio outstanding/loan count across different loan categories (Corporate, Mid-Corporate, SME) MUST be matched directly to VQ2.
   - Note: Questions asking for top 10 individual loan exposures or borrower level exposures MUST be matched to VQ5.
2. ROUTE TO GENERATED: If the question requires a specific filter, different calculations, or doesn't match any pre-approved template, select \"route\": \"generated\" and \"query_id\": null.
3. ROUTE TO ESCALATE: Select \"route\": \"escalate\" and \"query_id\": null if the user question requests:
   - Any restricted personal identifiable data (PII) including physical home addresses, personal/private phone numbers, or personal/private emails of bank employees, officers, or relationship managers.
   - Compensation, personal performance details, or private HR metadata.
   - Any other information entirely outside the scope of read-only commercial credit risk analytics.
   *(CRITICAL: Any question asking for physical home addresses or personal emails/contact details of relationship managers MUST be routed to \"escalate\" immediately)*

### USER QUESTION
{user_question}

### OUTPUT FORMAT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated" or "escalate",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text, markdown blocks, or explanations.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### ROLE
You are an expert SQL developer specializing in credit risk and portfolio analytics on a SQLite database.

### DATABASE SCHEMA
{schema_context}

### BUSINESS RULES
- NPA Definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
- Latest Reporting Date: 2025-09-30
- Latest Rating Date: 2025-09-30

### CONSTRAINTS
1. Use only tables and columns from the schema context. Never reference columns that do not exist.
2. Ensure the query is SQLite compatible. Never use functions like DATE_SUB or perform simple date subtraction (e.g., date1 - date2). For date differences in days, use: julianday(date1) - julianday(date2).
3. When asked for downgraded borrowers in the latest rating cycle, query the `borrower_rating` table for the latest rating_date ('2025-09-30') and rating_direction = 'Downgraded'.
4. Never return sensitive personal data such as home addresses or personal emails. If requested, write a harmless select query returning only relation manager names to let validation escalate appropriately.
5. Output only valid SQL. No explanations, no markdown formatting, no comments.

### USER QUESTION
{user_question}

### OUTPUT
Return ONLY the SQLite query string.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
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
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result

    # Robust multi-statement check: strip all trailing semicolons and whitespace, then check if any semicolon remains
    cleaned_sql = candidate_sql.strip().rstrip(';').strip()
    if ';' in cleaned_sql:
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    # NOTE: kept exactly as in the notebook. `unknown` is computed but is not used
    # to fail the query (aliases such as total_outstanding_million would be flagged).
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
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
You are a senior commercial lending and credit-risk SQL validator at
Northbridge Bank. Your responsibility is to determine whether a candidate
SQL query correctly answers the business user's question.

### VALIDATION CONTEXT
{track_context}

### USER QUESTION
{user_question}

### CANDIDATE SQL
{candidate_sql}

### VALIDATION INSTRUCTIONS
Evaluate the business relevance of the candidate SQL. The SQL has already
passed separate read-only, schema, and syntax checks, so focus on whether its
logic correctly answers the question.

Consider all of the following:

1. INTENT ALIGNMENT
   - Does the SQL address the metric or analysis requested by the user?
   - Examples include portfolio outstanding, sector exposure, overdue
     accounts, DPD aging, NPA exposure, credit-rating changes, restructuring,
     IFRS 9 stage, ECL, and provision coverage.

2. TABLES, JOINS, AND COLUMNS
   - Does the query use the tables and columns appropriate for the requested
     commercial-lending analysis?
   - Are joins logically appropriate and unlikely to duplicate or omit
     borrower, loan, rating, or provisioning records?

3. BUSINESS DEFINITIONS
   - Does the query implement the requested credit-risk definition correctly?
   - Check definitions involving overdue status, DPD buckets, NPA status,
     outstanding exposure, loan category, borrower or business-group
     exposure, IFRS 9 stage, ECL, and provision coverage ratio.

4. FILTERS AND REPORTING PERIOD
   - Does the query apply the required borrower, sector, category, status,
     threshold, date, quarter, or latest-reporting-period filter?
   - For a generated query, reject it when an essential requested filter is
     missing or incorrect.
   - For a verified template, follow the validation-track guidance above and
     do not reject it merely because it returns a complete result set.

5. AGGREGATION AND RESULT GRAIN
   - Are SUM, COUNT, AVG, grouping, ordering, ranking, and limits used
     correctly?
   - Does the query return results at the requested level, such as sector,
     borrower, business group, loan category, DPD bucket, IFRS 9 stage,
     rating, or reporting quarter?
   - If the user requests a top-N result, does the query order and limit the
     results correctly?

6. ANSWER SHAPE
   - Would the returned rows contain enough information to produce a clear
     and accurate answer to the user's question?
   - Do not require extra columns that are not necessary to answer the
     question.

### DECISION RULES
- Return "yes" only when the SQL is relevant and its business logic is
  reasonably capable of answering the user's question.
- Return "no" when the SQL uses the wrong metric, grain, aggregation,
  business rule, reporting period, entity, filter, ordering, or limit.
- For verified templates, allow a broader result set when the response
  generation step can select the relevant rows.
- Base the decision only on the user question, candidate SQL, and validation
  context provided.
- Confidence must be a number from 0.0 to 1.0.
- The reason must be one concise sentence explaining the most important
  basis for the verdict.
- Do not include Markdown, code fences, or text outside the JSON object.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';').strip()
        cleaned_candidate = candidate_sql.strip().rstrip(';').strip()

        # Helper: strip existing limit clauses to avoid 'near LIMIT: syntax error' when we append LIMIT 0
        def clean_limit(sql_str):
            return re.sub(r'\s+LIMIT\s+\d+\s*$', '', sql_str, flags=re.IGNORECASE).strip()

        expected_sql_for_desc = clean_limit(expected_sql)
        candidate_sql_for_desc = clean_limit(cleaned_candidate)

        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql_for_desc} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql_for_desc} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer specializing in SQLite and commercial
lending credit-risk data. Your task is to correct a SQL query that failed
validation.


### INSTRUCTIONS
1. Identify and fix the issue described in the validation error.
2. Preserve the original intent of the user's question.
3. Return a single read-only query beginning with SELECT or WITH.
4. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE,
   TRUNCATE, ATTACH, or any other operation that modifies the database.
5. Use only the tables and columns provided in the database schema.
6. Ensure all table aliases, column references, joins, filters,
   aggregations, groupings, and ordering clauses are valid.
7. Ensure the query uses the correct commercial-lending and credit-risk
   business logic for the user's question.
8. Ensure the corrected query is fully compatible with SQLite.
9. Do not add assumptions that are unsupported by the user question or
   database schema.
10. Return only the corrected SQL query, without Markdown code fences,
    comments, explanations, labels, or introductory text.


### INPUT
User Question:
{user_question}


Failed SQL:
{failed_sql}


Validation Error:
{error_message}


Database Schema:
{schema_context}


### OUTPUT
Return ONLY the corrected SQL query.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### ROLE
You are a commercial lending credit-risk analyst at Northbridge Bank.
Your task is to write a concise and accurate business response using only
the supplied query result.


### CONTEXT
Query Route: {route}

Verified Query ID:
{query_id if query_id is not None else "Not applicable"}


### INPUT
User Question:
{user_question}


Query Result Data:
{dataframe.to_string(index=False)}


### INSTRUCTIONS
1. Answer the user's specific question directly. Do not reproduce the entire
   result table.
2. Use only values present in the query result. Do not invent, estimate, or
   assume facts that are not shown.
3. Focus on the relevant sector, borrower, business group, loan category,
   DPD bucket, credit rating, IFRS 9 stage, or reporting quarter requested
   by the user.
4. Include rankings or portfolio comparisons only when they help answer the
   question.
5. State financial values using the unit indicated by the column name.
   Preserve the scale and precision shown in the result.
6. State percentage or ratio values as percentages only when the relevant
   source column represents a percentage.
7. Clearly distinguish outstanding exposure, ECL, provision amounts,
   borrower counts, account counts, DPD values, and percentage measures.
8. Mention the reporting date or quarter when it appears in the result and
   is relevant to the answer.
9. Highlight material observations supported by the data, such as exposure
   concentration, high delinquency, elevated NPA exposure, rating
   downgrades, weak provision coverage, or an increasing ECL trend.
10. If the query result is empty, state that no matching records were found
    in the available portfolio data.
11. Keep the response focused and professional. Use two to four sentences
    for a simple question and one short paragraph for a complex question.
12. Do not mention SQL, prompts, language models, internal validation, or
    processing steps.


### OUTPUT
Return ONLY the natural-language response, with no heading, code block,
Markdown table, or introductory label.
"""

    narrative = llm.invoke(response_prompt).content.strip()

    return narrative


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    The step-by-step trace that the notebook printed is collected in log['trace'].
    If verbose is True the trace lines are also echoed to the server console.
    '''
    start_time = time.time()
    trace = []

    def say(message):
        trace.append(message)
        if verbose:
            print(message)

    log = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'warnings': [],
        'trace': trace,
        'elapsed_seconds': None
    }

    def finish(df):
        log['elapsed_seconds'] = round(time.time() - start_time, 2)
        return {'log': log, 'dataframe': df, **log}

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Immediate escalation if routed to escalate
    if log['route'] == 'escalate':
        log['escalated'] = True
        log['narrative'] = "Query could not be reliably resolved. Escalated to human analyst. Reason: Restricted personal identification information request."
        log['confidence'] = 'ESCALATED'
        say("[!] Escalated immediately based on routing rules.")
        return finish(None)

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return finish(None)

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    say(f"[4] Execute: {len(df)} rows returned")
    for warning_text in exec_result['warnings']:
        say(f"    Warning: {warning_text}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative
    log['confidence'] = gate.get('relevance_confidence')

    return finish(df)


# =============================================================================
# EVALUATION AGAINST GROUND TRUTH (same metrics as the notebook)
# =============================================================================


def build_evaluation(test_results):
    evaluation_rows = []
    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results[i]
        evaluation_rows.append({
            'Test Case': gt['test_id'],
            'Expected Route': gt['expected_route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['expected_route'],
            'Expected Query ID': gt['expected_query_id'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (pd.isna(gt['expected_query_id']) and pd.isna(tr['query_id'])) or tr['query_id'] == gt['expected_query_id'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count'] if tr['row_count'] is not None else 0
        })
    evaluation_df = pd.DataFrame(evaluation_rows)
    path_accuracy = evaluation_df['Route Match'].mean() * 100
    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100
    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()
    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# =============================================================================
# UI HELPERS (no business logic below this line except presentation)
# =============================================================================


def safe_run(user_question):
    """Runs the pipeline and converts unexpected errors into a visible error result."""
    try:
        return run_pipeline(user_question, conn, verified_query_library, database_schema, verbose=False)
    except Exception as exc:  # LLM/API failure, malformed JSON from an LLM, SQLite error, and so on
        log = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'user_question': user_question,
            'route': 'error',
            'query_id': None,
            'match_reason': None,
            'candidate_sql': None,
            'gate_result': None,
            'retry_used': False,
            'escalated': True,
            'executed_sql': None,
            'row_count': None,
            'confidence': 'ERROR',
            'narrative': f"The pipeline stopped with an error: {type(exc).__name__}: {exc}",
            'warnings': [],
            'trace': [f"[!] Pipeline error: {type(exc).__name__}: {exc}"],
            'elapsed_seconds': None
        }
        return {'log': log, 'dataframe': None, **log}


def md_safe(text):
    """Escape dollar signs so Streamlit does not render amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def pretty_sql(sql):
    try:
        return sqlparse.format(sql, reindent=True, keyword_case='upper')
    except Exception:
        return sql


def format_confidence(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}"
    return "N/A" if value is None else str(value)


def record_audit(result, source="ad-hoc"):
    """Adds one audit entry to the session log and, if AUDIT_LOG_PATH is set, to a JSONL file."""
    gate = result.get('gate_result') or {}
    entry = {
        'timestamp_utc': result.get('timestamp_utc'),
        'source': source,
        'user_question': result.get('user_question'),
        'route': result.get('route'),
        'query_id': result.get('query_id'),
        'match_reason': result.get('match_reason'),
        'retry_used': result.get('retry_used'),
        'escalated': result.get('escalated'),
        'validation_passed': gate.get('passed'),
        'failed_check': gate.get('failed_check'),
        'validation_details': gate.get('details'),
        'candidate_sql': result.get('candidate_sql'),
        'executed_sql': result.get('executed_sql'),
        'row_count': result.get('row_count'),
        'confidence': result.get('confidence'),
        'elapsed_seconds': result.get('elapsed_seconds'),
        'narrative': result.get('narrative'),
    }
    st.session_state.audit_log.append(entry)

    audit_path = get_setting("AUDIT_LOG_PATH")
    if audit_path:
        try:
            with open(audit_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass


ROUTE_LABELS = {
    'verified': 'Verified template',
    'generated': 'Generated SQL',
    'escalate': 'Escalated',
    'error': 'Pipeline error',
}


def render_result(result, key_prefix):
    """Shows the answer, confidence, SQL used, raw data and the audit details for one run."""
    df = result['dataframe']

    if result['route'] == 'error':
        st.error(result['narrative'])
    elif result['escalated']:
        st.warning("Escalated to a human analyst")
        st.write(md_safe(result['narrative']))
    else:
        st.subheader("Answer")
        st.markdown(md_safe(result['narrative']))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Route", ROUTE_LABELS.get(result['route'], str(result['route'])))
    col2.metric("Verified query ID", result['query_id'] or "n/a")
    col3.metric("Confidence score", format_confidence(result['confidence']))
    col4.metric("Rows returned", result['row_count'] if result['row_count'] is not None else 0)

    if result['match_reason']:
        st.caption(f"Routing reason: {result['match_reason']}")

    sql_to_show = result['executed_sql'] or result['candidate_sql']
    if sql_to_show:
        st.markdown("**SQL used**" if result['executed_sql'] else "**SQL attempted (not executed)**")
        st.code(pretty_sql(sql_to_show), language="sql")

    if df is not None:
        st.markdown("**Raw data returned**")
        st.dataframe(df)
        st.download_button(
            "Download result as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{result['query_id'] or 'generated'}_result.csv",
            mime="text/csv",
            key=f"download_{key_prefix}",
        )

    for warning_text in result.get('warnings') or []:
        st.warning(warning_text)

    with st.expander("Validation gate and pipeline trace"):
        gate = result.get('gate_result')
        if gate:
            st.json(gate)
        st.text("\n".join(result.get('trace') or []))
        st.caption(
            f"Retry used: {result['retry_used']} | "
            f"Elapsed: {result['elapsed_seconds']} seconds | "
            f"Timestamp (UTC): {result['timestamp_utc']}"
        )


# =============================================================================
# RUNTIME INITIALISATION (credentials, database, LLMs)
# =============================================================================

api_key_found, api_base_found = configure_credentials()
db_file = resolve_db_path()
db_found = db_file.exists()

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

with st.sidebar:
    st.header("Credit Risk Query Engine")
    st.write(
        "Ask routine commercial lending portfolio questions in plain English. "
        "Recurring questions use pre-approved SQL templates. Other questions use "
        "generated read-only SQL that must pass a validation gate."
    )

    st.subheader("Safeguards")
    st.markdown(
        "- Read-only database connection (SQLite `mode=ro`)\n"
        "- Only `SELECT` / `WITH` statements are allowed\n"
        "- Validation: read-only shape, schema read, parse and plan, LLM relevance, template integrity\n"
        "- One retry for generated SQL, then escalation to a human analyst\n"
        "- SQL, raw data and confidence score shown with every answer"
    )

    st.subheader("Reporting metadata")
    st.markdown(
        "- Latest provisioning date: 2025-09-30\n"
        "- Latest rating date: 2025-09-30\n"
        "- NPA: `Substandard`, `Doubtful`, `Loss`"
    )

    st.subheader("Configuration status")
    st.write(f"OpenAI API key: {'found' if api_key_found else 'missing'}")
    st.write(f"OpenAI API base URL: {'found' if api_base_found else 'not set (library default)'}")
    st.write(f"Database file: {'found' if db_found else 'missing'}")
    st.caption(f"Database path: {db_file}")
    st.caption("Models: gpt-4o-mini (routing, SQL, narrative), gpt-4o (relevance validation)")

if not api_key_found:
    st.error(
        "OPENAI_API_KEY is not configured. Set it as an environment variable, in "
        ".streamlit/secrets.toml (or the Streamlit Cloud Secrets panel), or in a local config.json."
    )
    st.stop()

if not db_found:
    st.error(
        f"Database file not found at {db_file}. Place credit_risk_portfolio.db next to app.py "
        "or set DB_PATH to its location."
    )
    st.stop()

conn = get_connection(str(db_file))
llm, evaluator_llm = init_llms()

# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================

st.title("Credit Risk Query Engine")
st.caption("Northbridge Bank | Commercial lending portfolio | Read-only analytics")

tab_ask, tab_lib, tab_eval, tab_audit, tab_db = st.tabs([
    "Ask a question",
    "Verified query library",
    "Evaluation",
    "Audit log",
    "Database and schema",
])

# ----------------------------------------------------------------------------
# Tab 1: Ask a question
# ----------------------------------------------------------------------------
with tab_ask:
    def use_example(question_text):
        st.session_state.question_input = question_text

    with st.expander("Example questions (from the test cases)"):
        for idx, example in enumerate(ground_truth['question'].tolist()):
            st.button(example, key=f"example_{idx}", on_click=use_example, args=(example,))

    st.text_area(
        "Your question",
        key="question_input",
        height=100,
        max_chars=1000,
        placeholder="For example: What is the total portfolio outstanding and loan count across different loan categories?",
    )

    if st.button("Run query", type="primary"):
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Routing, validating and running your question..."):
                result = safe_run(question)
            st.session_state.last_result = result
            record_audit(result, source="ad-hoc")

    if st.session_state.last_result is not None:
        st.divider()
        render_result(st.session_state.last_result, key_prefix="ask")

# ----------------------------------------------------------------------------
# Tab 2: Verified query library
# ----------------------------------------------------------------------------
with tab_lib:
    st.subheader("Verified query template library")
    st.write(
        f"{len(verified_query_library)} pre-approved SQL templates. Each runs without modification; "
        "the narrative is written from the full result."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(pretty_sql(entry['sql']), language="sql")
            if st.button(f"Run {qid} directly", key=f"run_{qid}"):
                try:
                    st.dataframe(pd.read_sql_query(entry['sql'], conn))
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")

# ----------------------------------------------------------------------------
# Tab 3: Evaluation against ground truth
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation against ground truth")
    st.write(
        "Runs the five test cases through the full pipeline and reports Selected Path Accuracy, "
        "Selected Query Accuracy and Average Confidence Score."
    )
    with st.expander("Ground-truth test cases"):
        st.dataframe(ground_truth)

    if st.button("Run all 5 test cases"):
        results = []
        progress = st.progress(0.0, text="Starting...")
        for i, question in enumerate(ground_truth['question'].tolist()):
            progress.progress(i / len(ground_truth), text=f"Running {ground_truth['test_id'].iloc[i]}...")
            res = safe_run(question)
            record_audit(res, source=f"evaluation:{ground_truth['test_id'].iloc[i]}")
            results.append(res)
        progress.progress(1.0, text="Done")
        st.session_state.eval_results = results

    if st.session_state.eval_results is not None:
        results = st.session_state.eval_results
        evaluation_df, path_accuracy, query_accuracy, average_confidence = build_evaluation(results)

        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
        m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
        m3.metric("Average Confidence Score", "N/A" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

        display_df = evaluation_df.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(format_confidence)
        display_df['Expected Query ID'] = display_df['Expected Query ID'].apply(lambda v: "n/a" if v is None else v)
        display_df['Actual Query ID'] = display_df['Actual Query ID'].apply(lambda v: "n/a" if v is None else v)
        st.dataframe(display_df)
        st.download_button(
            "Download evaluation as CSV",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name="evaluation_results.csv",
            mime="text/csv",
            key="download_evaluation",
        )

        st.markdown("**Test case details**")
        for i, res in enumerate(results):
            test_id = ground_truth['test_id'].iloc[i]
            with st.expander(f"{test_id}: {ground_truth['question'].iloc[i]}"):
                st.caption(f"Expected: {ground_truth['expected_answer'].iloc[i]}")
                render_result(res, key_prefix=f"eval_{i}")

# ----------------------------------------------------------------------------
# Tab 4: Audit log
# ----------------------------------------------------------------------------
with tab_audit:
    st.subheader("Audit log (this session)")
    st.caption(
        "Every run in this session is recorded. To keep a permanent record, set AUDIT_LOG_PATH "
        "so each entry is also appended to a JSONL file, or download the log below."
    )
    if not st.session_state.audit_log:
        st.info("No queries have been run in this session yet.")
    else:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        audit_df['confidence'] = audit_df['confidence'].apply(format_confidence)
        st.dataframe(audit_df)
        col_a, col_b = st.columns(2)
        col_a.download_button(
            "Download audit log (CSV)",
            data=audit_df.to_csv(index=False).encode('utf-8'),
            file_name="audit_log.csv",
            mime="text/csv",
            key="download_audit_csv",
        )
        col_b.download_button(
            "Download audit log (JSON)",
            data=json.dumps(st.session_state.audit_log, indent=2, default=str).encode('utf-8'),
            file_name="audit_log.json",
            mime="application/json",
            key="download_audit_json",
        )

# ----------------------------------------------------------------------------
# Tab 5: Database and schema
# ----------------------------------------------------------------------------
with tab_db:
    st.subheader("Database schema provided to the LLM")
    st.code(database_schema)

    st.subheader("Table preview (3 rows)")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
    table_name = st.selectbox("Table", tables['name'].tolist())
    if table_name:
        row_count = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{table_name}"', conn)['n'].iloc[0]
        st.caption(f"{table_name}: {row_count:,} rows")
        st.dataframe(pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 3', conn))

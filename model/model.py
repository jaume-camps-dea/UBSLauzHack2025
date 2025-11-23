try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.runnables import Runnable
except Exception:  
    ChatVertexAI = None  
    ChatPromptTemplate = None  
    JsonOutputParser = None  
    StrOutputParser = None  
    class Runnable:  
        pass

import pandas as pd
from google.cloud import bigquery, bigquery_storage
from google.oauth2 import service_account
import json
import re

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any


class Text_2_Sql_Output(BaseModel):
    sql_query: str
    involve_multiple_users: bool

class SummarizerOutput(BaseModel):
    summary: str

def get_user_info(partner_id):
    query = f"""

    WITH base_partner AS (
    SELECT 
        p.partner_id,
        p.partner_name,
        p.partner_gender,
        p.partner_birth_year,
        p.partner_phone_number,
        p.partner_address,
        p.industry_gic2_code,
        p.partner_class_code
    FROM `production_dataset.partner` p
    WHERE p.partner_id = '{partner_id}'
    AND p.partner_class_code != 'S'
),

active_roles AS (
    SELECT 
        pr.partner_id,
        pr.entity_id AS br_id
    FROM `production_dataset.partner_role` pr
    WHERE pr.entity_type = 'BR'
    AND pr.relationship_end_date IS NULL
),

active_brs AS (
    SELECT
        br.br_id,
        br.br_open_date,
        br.br_close_date
    FROM `production_dataset.business_rel` br
    WHERE br.br_close_date IS NULL
),

partner_brs AS (
    SELECT
        ar.partner_id,
        ARRAY_AGG(STRUCT(
            abr.br_id,
            abr.br_open_date,
            abr.br_close_date
        )) AS business_relationships
    FROM active_roles ar
    JOIN active_brs abr ON ar.br_id = abr.br_id
    GROUP BY ar.partner_id
),

active_account_links AS (
    SELECT 
        bta.br_id,
        bta.account_id
    FROM `production_dataset.br_to_account` bta
    WHERE bta.relationship_status_code = 1
),

active_accounts AS (
    SELECT
        a.account_id,
        a.account_iban,
        a.account_currency,
        a.account_open_date,
        a.account_close_date
    FROM `production_dataset.account` a
    WHERE a.account_close_date IS NULL
),

partner_accounts AS (
    SELECT 
        ar.partner_id,
        ARRAY_AGG(STRUCT(
            aa.account_id,
            aa.account_iban,
            aa.account_currency,
            aa.account_open_date,
            aa.account_close_date
        )) AS accounts
    FROM active_roles ar
    JOIN active_account_links al ON ar.br_id = al.br_id
    JOIN active_accounts aa ON al.account_id = aa.account_id
    GROUP BY ar.partner_id
),

account_transactions AS (
    SELECT
        t.account_id,
        ARRAY_AGG(STRUCT(
            t.account_id AS performing_account_id,  -- <--- added account_id here
            t.debit_credit,
            CAST(t.amount AS NUMERIC) AS amount,
            CAST(t.balance AS NUMERIC) AS balance,
            t.currency,
            t.transaction_date,
            t.transfer_type,
            t.counterparty_account_id,
            t.ext_counterparty_account_id,
            t.ext_counterparty_country
        ) ORDER BY t.transaction_date DESC) AS transactions
    FROM `production_dataset.transactions` t
    GROUP BY t.account_id
),

partner_transactions AS (
    SELECT
        ar.partner_id,
        ARRAY_CONCAT_AGG(tx.transactions) AS all_transactions
    FROM active_roles ar
    JOIN active_account_links al ON ar.br_id = al.br_id
    JOIN account_transactions tx ON al.account_id = tx.account_id
    GROUP BY ar.partner_id
),

-- NEW: Add partner notes CTE
partner_notes AS (
    SELECT
        con.partner_id,
        ARRAY_AGG(STRUCT(
            con.onboarding_note
        )) AS onboarding_notes
    FROM `production_dataset.client_onboarding_notes` con
    WHERE con.partner_id = '{partner_id}'
    GROUP BY con.partner_id
)

SELECT
    bp.*,
    br.business_relationships,
    ac.accounts,
    tr.all_transactions,
    pn.onboarding_notes  -- NEW: Include onboarding notes in final result
FROM base_partner bp
LEFT JOIN partner_brs br ON bp.partner_id = br.partner_id
LEFT JOIN partner_accounts ac ON bp.partner_id = ac.partner_id
LEFT JOIN partner_transactions tr ON bp.partner_id = tr.partner_id
LEFT JOIN partner_notes pn ON bp.partner_id = pn.partner_id;  -- NEW: Join with notes table


    """
    return query


def pipeline(user_input,client):

    llm = ChatVertexAI(
        model_name='gemini-2.0-flash-001',
        project='dea-analysis',
        location='us-central1',
        temperature=0,
    )
    

    text_to_sql_parser = JsonOutputParser(pydantic_object=Text_2_Sql_Output)

    text_to_sql_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    You are a Senior Data Engineer at UBS. Your task is to translate natural language questions into precise, optimized, and executable Google BigQuery (Standard SQL) queries. Follow the database schema, join paths, and business logic strictly.

    Database Configuration:

    Project/Dataset: production_dataset
    All tables must be fully qualified with this prefix.

    Syntax: Google BigQuery Standard SQL.
    Identifiers: Always wrap table and column names in backticks (`).

    Schema & Business Logic:
    1. partner:
    Keys: partner_id
    Columns: industry_gic2_code, partner_class_code, partner_gender, partner_name,
                partner_phone_number, partner_birth_year, partner_address,
                partner_open_date, partner_close_date
    Rules:
    - partner_class_code = 'S' are synthetic. Exclude unless explicitly requested.
    - Use ILIKE for name search.

    2. partner_country:
    Keys: partner_id
    Columns: country_name, partner_country_status_code, country_type
    Rules:
    - Default to country_type = 'domicile' AND partner_country_status_code = 1

    3. partner_role:
    Keys: partner_id, entity_id, entity_type
    Rules:
    - entity_type = 'BR'
    - relationship_end_date IS NULL → active
    - br_type_code: VP=Individual, JO=Joint
    - associated_partner_id: joint/beneficial ownership mapping

    4. business_rel:
    Keys: br_id
    Rules:
    - br_close_date IS NULL → active

    5. br_to_account:
    Keys: br_id, account_id
    Rules:
    - relationship_status_code = 1 → active

    6. account:
    Keys: account_id
    Rules:
    - account_close_date IS NULL → active

    7. transactions:
    Rules:
    - credit = money IN
    - debit = money OUT
    - net flow = SUM(CASE WHEN debit_credit='credit' THEN amount ELSE -amount END)
    - always ROUND(CAST(amount AS NUMERIC), 2)
    - if ext_counterparty_account_id is null, then means that the transaction has been made between inside the bank. If it has a value, then the transaction was to/from outside the company (external)

    8. client_onboarding_notes:
    Free text search when relevant.

    Golden Join Path
    --------------------------------------------------
    partner
    → partner_role
    → business_rel
    → br_to_account
    → account
    → optional: transactions

    Active Filters (default)
    --------------------------------------------------
    partner_role.relationship_end_date IS NULL
    business_rel.br_close_date IS NULL
    br_to_account.relationship_status_code = 1
    account.account_close_date IS NULL
    partner_country.partner_country_status_code = 1 (if used)

    CONSIDERATIONS
    - Apart from the information required, add the partner_id field so we can know the partner related to the request.
    eg. If they ask for the name of the client with most transactions, we need to return the query to extract the client name, but also its partner_id
    - If we return an amount of money, return the currency as well!
    - If the query envolves one single user (that is, we are just looking for a concreate user, instead or a group of them), involve_multiple_users must be False. Otherwise, True
    Formatting Requirements
    {format_instructions}
    """
        ),
        (
            "human",
            """
    Given the following user request, generate the appropriate BigQuery SQL query following all rules and business logic:
    {user_request}
    """
        )
    ]).partial(
        format_instructions=text_to_sql_parser.get_format_instructions() # format_instructions inludes all the output format with the explanation (how to perform the output, based in AnalysisOutput class)
    )

    chain = text_to_sql_prompt | llm | text_to_sql_parser
    result1 = chain.invoke({
        "user_request": user_input
    })

    if result1["involve_multiple_users"]:
        return {"result":{},"summary": "The query involves multiple Users. The question can be answered by other llm", "involve_multiple_users": result1["involve_multiple_users"]}
    
    query_job = client.query(result1["sql_query"])
    results = [dict(row) for row in query_job][0]

    summarizer_parser = JsonOutputParser(pydantic_object=SummarizerOutput)
    summarizer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    You are a summarizer.
    Based on the user query and the results obtained, explain the results. 
    
    Formatting Requirements
    {format_instructions}
    """
        ),
        (
            "human",
            """
    Given the following user request, generate the appropriate summary
    {user_request}
    Here the data obtained:
    {results}
    """
        )
    ]).partial(
        format_instructions=summarizer_parser.get_format_instructions() # format_instructions inludes all the output format with the explanation (how to perform the output, based in AnalysisOutput class)
    )

    chain = summarizer_prompt | llm | summarizer_parser
    result2 = chain.invoke({
        "user_request": user_input,
        "results": results
    })


    user_info_query = get_user_info(results["partner_id"])


    query_job = client.query(user_info_query)
    results = [dict(row) for row in query_job][0]


    return {"result":results, "summary": result2["summary"], "involve_multiple_users": False}
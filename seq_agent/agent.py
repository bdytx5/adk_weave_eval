




import json
from typing import Any, Dict, List, Optional

# ---- Load data ----
with open("insurancedata.json", "r", encoding="utf-8") as f:
    INSURANCE_DATA = json.load(f)

def policy_lookup(policy_id: str) -> Optional[Dict[str, Any]]:
    return next((p for p in INSURANCE_DATA['policy_lookup'] if p['policy_id'] == policy_id), None)

def customer_policies(customer_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        p for p in INSURANCE_DATA['policy_lookup']
        if p['customer_id'] == customer_id and (status is None or p['status'] == status)
    ]

def claims_status_checker(policy_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        c for c in INSURANCE_DATA['claims_status_checker']
        if c['policy_id'] == policy_id and (status is None or c['status'] == status)
    ]

def coverage_calculator(policy_id: str, option: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        cc for cc in INSURANCE_DATA['coverage_calculator']
        if cc['policy_id'] == policy_id and (option is None or cc['option'] == option)
    ]

def premium_payment_system(policy_id: str) -> List[Dict[str, Any]]:
    return [
        i for i in INSURANCE_DATA['premium_payment_system']
        if i['policy_id'] == policy_id
    ]

def appointment_schedule_checker(customer_id: str) -> List[Dict[str, Any]]:
    return [
        a for a in INSURANCE_DATA['appointment_schedule_checker']
        if a['customer_id'] == customer_id
    ]

def faq_search(query: str, topk: int = 1) -> List[Dict[str, Any]]:
    matches = [f for f in INSURANCE_DATA['faq_search'] if query.lower() in f['question'].lower()]
    return matches[:topk]

def find_nearby_repair_shop(zip_code: str, approved_only: bool = True, topk: int = 3) -> List[Dict[str, Any]]:
    shops = [s for s in INSURANCE_DATA['find_nearby_repair_shop'] if s['zip'] == zip_code]
    if approved_only:
        shops = [s for s in shops if s['approved']]
    return shops[:topk]

def insurance_quote_data(customer_id: str) -> List[Dict[str, Any]]:
    return [
        q for q in INSURANCE_DATA['insurance_quote_data']
        if q['customer_id'] == customer_id
    ]


agent_prompt = """
policy_lookup(policy_id)
# Use when you need all details for a specific policy using its policy_id (such as viewing coverages, start/end, type, etc).
print(policy_lookup("P100001"))
# Returns:
# {'policy_id': 'P100001', 'customer_id': 'C20001', ..., 'status': 'active'}

customer_policies(customer_id, status=None)
# Use when you want all policies (optionally filtered by status, e.g. 'active') for a given customer_id.
print(customer_policies("C20001", status="active"))
# Returns:
# [{'policy_id': 'P100001', ...}]

claims_status_checker(policy_id, status=None)
# Use when you need all claims for a given policy. Optionally filter by claim status ('pending', 'approved', etc).
print(claims_status_checker("P100001", status="pending"))
# Returns:
# [{'claim_id': 'CL5001', 'policy_id': 'P100001', 'status': 'pending', ...}]

coverage_calculator(policy_id, option=None)
# Use when you want to see alternate coverage/premium options for a policy, like how much premium changes if you change something.
print(coverage_calculator("P100001"))
# Returns:
# [{'policy_id': 'P100001', 'option': 'raise liability', 'change': 25000, 'new_premium': 103.50, 'new_deductible': 500}]

premium_payment_system(policy_id)
# Use when you need to see billing or payment info for a policy (invoices, due dates, amounts).
print(premium_payment_system("P100001"))
# Returns:
# [{'invoice_id': 'INV3001', 'policy_id': 'P100001', 'due_date': '2023-05-05', 'amount_due': 120.00, 'status': 'unpaid'}]

appointment_schedule_checker(customer_id)
# Use to list all scheduled or completed appointments (like adjuster visits or callbacks) for a customer.
print(appointment_schedule_checker("C20001"))
# Returns:
# [{'appointment_id': 'A4101', 'customer_id': 'C20001', 'type': 'callback', 'date': '2023-05-06T10:00:00', 'status': 'scheduled'}]

faq_search(query, topk=1)
# Use to look up answers to common insurance questions by keyword.
print(faq_search("deductible", topk=2))
# Returns:
# [{'faq_id': 'F002', 'question': 'What is a deductible?', 'answer': ...},
#  {'faq_id': 'F007', ...}]

find_nearby_repair_shop(zip_code, approved_only=True, topk=3)
# Use when you want to find repair shops near a ZIP code, optionally only approved ones. Useful for auto claims.
print(find_nearby_repair_shop("60636"))
# Returns:
# [{'shop_id': 'R01', 'name': 'AutoFix Pros', ...},
#  {'shop_id': 'R02', ...}]

insurance_quote_data(customer_id)
# Use when you wish to view all (recent/past) insurance quotes for a customer.
print(insurance_quote_data("C20001"))
# Returns:
# [{'quote_id': 'Q3001', 'customer_id': 'C20001', 'coverage': 'auto-basic', ...}]
"""

from google.adk.agents import Agent
import os

# Configure environment if needed
os.environ["GOOGLE_CLOUD_PROJECT"] = "dsports-6ab79"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"  # or set as needed

root_agent = Agent(
    model="gemini-2.5-pro",
    name="insurance_agent",
    instruction="""
You are an expert insurance assistant. You can answer customer questions and perform insurance-related tasks
using the available tools. For every question, carefully select and use the most relevant tool(s) from your toolkit.
Never guess—return only what is found in the database or tools. Here are your tool instructions and usage examples:

""" + agent_prompt,
    description="An insurance assistant that can answer any insurance query using the provided tools.",
    tools=[
        policy_lookup,
        customer_policies,
        claims_status_checker,
        coverage_calculator,
        premium_payment_system,
        appointment_schedule_checker,
        faq_search,
        find_nearby_repair_shop,
        insurance_quote_data
    ]
)
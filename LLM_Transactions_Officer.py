import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

#==============================================================================
# Configuration
#==============================================================================
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")

device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_model(name, token):
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        use_auth_token=token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        use_auth_token=token,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, generator

# Load model on first use
tokenizer, generator = load_model(MODEL_NAME, HUGGINGFACE_TOKEN)

#==============================================================================
# Prompt Template
#==============================================================================
PROMPT_TEMPLATE = (
    "You are an expert Anti-Fraud Officer assistant."
    "\nGiven the following transaction details, provide:"  
    "\n1. A concise one-sentence summary."
    "\n2. Key risk indicators."  
    "\n3. Contextual interpretation of possible fraud patterns."  
    "\nTransaction details:"  
    "\n{row}\n"
)

# Required transaction fields
REQUIRED_FIELDS = [
    "transaction_id", "amount", "location", "device", "merchant", "timestamp"
]

#==============================================================================
# Core Functions
#==============================================================================

def describe_transaction(details: dict, max_tokens: int = 256) -> str:
    """
    Generate a natural-language description for a single transaction using the open-source LLM.
    """
    # Format details; skip empty values
    details_str = "\n".join(
        [f"- {k}: {v}" for k, v in details.items() if v not in (None, "")]
    )
    prompt = PROMPT_TEMPLATE.format(row=details_str)

    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs[0]["generated_text"].strip()

#==============================================================================
# Streamlit App
#==============================================================================
st.set_page_config(page_title="Transaction Describer", layout="centered")
st.title("💼 Transaction Describer & Anti-Fraud Assistant")

st.write(
    "Enter transaction details below. The system will generate a summary, highlight risk indicators, and provide contextual interpretation."
)

with st.form(key="transaction_form"):
    form_data = {}
    form_data["transaction_id"] = st.text_input("Transaction ID")
    form_data["amount"] = st.number_input("Amount", min_value=0.0, format="%.2f")
    form_data["location"] = st.text_input("Location (e.g., city, country)")
    form_data["device"] = st.text_input("Device (e.g., iOS app, Web)")
    form_data["merchant"] = st.text_input("Merchant Name")
    form_data["timestamp"] = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)")
    form_data["account_age_days"] = st.number_input("Account Age (days)", min_value=0)
    submitted = st.form_submit_button("Describe Transaction")

if submitted:
    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if not form_data.get(f)]

    description = describe_transaction(form_data)

    if missing:
        st.warning(f"Missing important information: {', '.join(missing)}")
    st.subheader("Generated Description")
    st.write(description)

# NOTE: For production, add rate limiting, caching of previous responses, and error handling around the LLM calls.

# utils.py
# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
                                                                                
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_serper_api_key():
    load_env()
    openai_api_key = os.getenv("SERPER_API_KEY")
    return openai_api_key

def get_hf_api_key():
    load_env()
    hf_api_key = os.getenv("HF_API_KEY")
    return hf_api_key
def get_groq_api_key():
    load_env()
    groq_api_key = os.getenv("GROQ_API_KEY") 
    return groq_api_key

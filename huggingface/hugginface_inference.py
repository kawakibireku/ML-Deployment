import requests
import streamlit as st

TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/w11wo/indonesian-roberta-base-sentiment-classifier"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json(), response.status_code
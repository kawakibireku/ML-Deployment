from dotenv import load_dotenv
import os
import requests
load_dotenv()
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
print(TOKEN)
API_URL = "https://api-inference.huggingface.co/models/w11wo/indonesian-roberta-base-sentiment-classifier"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json(), response.status_code
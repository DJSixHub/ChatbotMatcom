import streamlit as st
import random
import time
import requests
import os
import streamlit as st
import json


class Chat:
    def __init__(self, model, api_key=None):
        self.headers = {
            "Authorization": f"Bearer {os.getenv("Token")}",
            "Content-Type": "application/json",
        }
        self.model = model
        if api_key is None:
            api_key = os.getenv(os.getenv("Token"))
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def run(self, context, input, history):
        promt_template = f"""
            Eres un asistente para la formación vocacional de la Universidad de La Habana. 
            Utiliza la siguiente información de referencia para enriquecer tu respuesta: {context}. 
            Historial de conversación: {history}. 
            El usuario ha indicado: {input}. 
            Con base en esta información, responde de manera clara, precisa y profesional, 
            orientando al usuario en su proceso de formación vocacional.
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"{promt_template}", "name": "User"}
            ],
        }

        response = requests.request(
            "POST", self.url, json=payload, headers=self.headers
        )

        return json.loads(response.text)


if __name__ == "__main__":
    chat = Chat("accounts/fireworks/models/llama-v3p1-8b-instruct", os.getenv("Token"))
    print(chat.run("que es cibernetica", "", [])["choices"][0]["message"])

import streamlit as st
import random
import time
import requests
import os


class Chat:
    def __init__(self, model, api_key=None):
        """
        Inicializa el cliente de Fireworks AI.

        Parámetros:
          - model: Nombre o identificador del modelo a usar.
          - api_key: Clave de API de Fireworks. Si no se proporciona, se busca en la variable de entorno FIREWORKS_API_KEY.
        """
        self.headers = {
            "Authorization": f"Bearer {os.getenv("Token")}",
            "Content-Type": "application/json",
        }
        self.model = model
        if api_key is None:
            api_key = os.getenv(os.getenv("Token"))
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"

    def run(self, prompt, historial):

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "<string>", "name": "<string>"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "description": "<string>",
                        "name": "<string>",
                        "parameters": {
                            "type": "object",
                            "required": ["<string>"],
                            "properties": {},
                        },
                    },
                }
            ],
            "max_tokens": 2000,
            "prompt_truncate_len": 1500,
            "temperature": 1,
            "top_p": 1,
            "top_k": 50,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "mirostat_lr": 0.1,
            "mirostat_target": 1.5,
            "n": 1,
            "ignore_eos": False,
            "stop": "<string>",
            "response_format": None,
            "stream": False,
            "context_length_exceeded_behavior": "truncate",
            "user": "<string>",
        }

        response = requests.request(
            "POST", self.url, json=payload, headers=self.headers
        )

        return response.text


# Ejemplo de uso:
if __name__ == "__main__":
    chat = Chat("accounts/fireworks/models/llama-v3p1-8b-instruct", os.getenv("Token"))
    print(chat.run("2+2", []))

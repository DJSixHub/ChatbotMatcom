import streamlit as st
import random
import time
import requests
import os
import streamlit as st
import json
import numpy as np


class Embedder:
    def __init__(
        self,
        md_file_path,
        chunk_size=10000,
        output_dir="/kaggle/working/chunks",
        output_prefix="chunk",
    ):
        """
        Inicializa la clase:
          - Lee y fragmenta el archivo Markdown.
          - Escribe cada chunk a un archivo.
          - Calcula el embedding de cada chunk mediante la API de Fireworks.
          - Almacena internamente dos arrays: uno de nombres de archivo y otro con los embeddings.

        Parámetros:
          md_file_path (str): Ruta al archivo Markdown.
          chunk_size (int): Tamaño máximo (en caracteres) para cada chunk.
          output_dir (str): Directorio en el que se escribirán los archivos de chunk.
          output_prefix (str): Prefijo para los nombres de archivo de los chunks.
        """
        if not os.path.exists(md_file_path):
            raise FileNotFoundError(
                f"Markdown file '{md_file_path}' does not exist. Please check the path."
            )

        self.md_file_path = md_file_path
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.output_prefix = output_prefix

        # Procesar el archivo Markdown
        self.chunks = self._chunk_markdown_file()
        print(f"Total chunks created: {len(self.chunks)}")

        # Procesar los chunks: guardarlos en archivos y calcular sus embeddings.
        self.embeddings_dict = self._process_chunks()

        # Convertir el diccionario de embeddings en dos arrays: file_names y embeddings_array.
        self.file_names, self.embeddings_array = self._convert_embeddings_dict()
        print(
            f"Embeddings computed for {len(self.file_names)} chunks. Embeddings array shape: {self.embeddings_array.shape}"
        )

    # ---------------------------
    # Función para obtener embedding
    # ---------------------------
    @staticmethod
    def get_embedding(text):
        """
        Obtiene el embedding para un texto dado usando la API de Fireworks.
        """
        url = "https://api.fireworks.ai/inference/v1/embeddings"
        payload = {
            "input": text,
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "dimensions": 768,
        }
        headers = {
            "Authorization": "Bearer fw_3ZR81bUKaAkvohmKYAmgycJg",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        embedding = response_json["data"][0]["embedding"]
        return embedding

    # ---------------------------
    # Función para fragmentar el Markdown en chunks
    # ---------------------------
    def _chunk_markdown_file(self):
        """
        Lee el archivo Markdown y lo fragmenta en chunks sin cortar párrafos.
        """
        with open(self.md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Se asume que los párrafos están separados por una o más líneas en blanco.
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = current_chunk + "\n\n" + para if current_chunk else para

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    # Caso especial: un párrafo mayor al chunk_size
                    chunks.append(para)
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # ---------------------------
    # Función para escribir chunks y calcular sus embeddings
    # ---------------------------
    def _process_chunks(self):
        """
        Escribe cada chunk a un archivo y obtiene su embedding.
        Retorna un diccionario con {ruta_del_archivo: embedding}.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        chunk_embeddings = {}

        for i, chunk in enumerate(self.chunks):
            filename = os.path.join(self.output_dir, f"{self.output_prefix}_{i+1}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chunk)
            print(f"Wrote chunk {i+1} to {filename}")

            try:
                embedding = Embedder.get_embedding(chunk)
                chunk_embeddings[filename] = embedding
                print(f"Obtained embedding for {filename} (length: {len(embedding)})")
            except Exception as e:
                print(f"Failed to obtain embedding for {filename}: {e}")
                chunk_embeddings[filename] = None

        return chunk_embeddings

    # ---------------------------
    # Función para convertir diccionario de embeddings a arrays
    # ---------------------------
    def _convert_embeddings_dict(self):
        """
        Convierte el diccionario de embeddings en dos arrays:
          - Lista de nombres de archivos.
          - Array numpy de embeddings.
        """
        file_names = []
        embedding_list = []
        for fname, emb in self.embeddings_dict.items():
            if emb is not None:
                file_names.append(fname)
                embedding_list.append(emb)
        embeddings_array = np.array(embedding_list)
        return file_names, embeddings_array

    # ---------------------------
    # Función para encontrar los top-k chunks más similares
    # ---------------------------
    def query_top_k(self, input_text, k=3):
        """
        Dado un texto de entrada, calcula su embedding y devuelve los top k chunks
        (índices, puntajes de similitud y nombres de archivo) más similares utilizando
        similitud coseno.

        Parámetros:
          input_text (str): Texto de consulta.
          k (int): Número de resultados a retornar.

        Retorna:
          tuple: (top_indices, top_scores, top_file_names)
        """
        # Calcular el embedding para el texto de entrada.
        input_embedding = np.array(Embedder.get_embedding(input_text))

        norm_input = np.linalg.norm(input_embedding)
        if norm_input == 0:
            raise ValueError("Input embedding has zero norm.")
        input_embedding_norm = input_embedding / norm_input

        # Normalizar los embeddings de los chunks.
        norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # evitar división por cero
        embeddings_norm = self.embeddings_array / norms

        # Calcular la similitud coseno.
        cosine_similarities = np.dot(embeddings_norm, input_embedding_norm)
        top_indices = np.argsort(cosine_similarities)[::-1][:k]
        top_scores = cosine_similarities[top_indices]
        top_file_names = [self.file_names[i] for i in top_indices]

        return top_indices, top_scores, top_file_names

    # ---------------------------
    # Método auxiliar para imprimir el contenido de un chunk dado su archivo
    # ---------------------------
    @staticmethod
    def file_contents(file_path):
        """
        Lee y muestra el contenido de un archivo de texto.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                return content
        except FileNotFoundError:
            print(f"El archivo '{file_path}' no se encontró.")
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")


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
        """self.e = Embedder(
            "MarkdownFile.md",
            chunk_size=10000,
            output_dir="output",
            output_prefix="chunk",
        )"""

    def run(self, context, input, history):
        with open("prompts.json", encoding="utf-8") as f:
            data = json.load(f)

        k = 3
        context = ""
        """top_indices, top_scores, top_file_names = self.e.query_top_k(input, k=k)
        for idx, score, fname in zip(top_indices, top_scores, top_file_names):
            context.join(self.e.file_contents(fname))"""

        promt_template = (
            f"{input}"
            + data["formacion_formacional"]["promt"]
            + data["formacion_formacional"]["history"]
            + f"{history}"
            + data["formacion_formacional"]["contexto"]
            + f"{context}"
            + data["formacion_formacional"]["end"]
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{promt_template}",
                    "name": "User",
                }
            ],
        }

        response = requests.request(
            "POST", self.url, json=payload, headers=self.headers
        )
        print(json.loads(response.text))
        return json.loads(response.text)


if __name__ == "__main__":
    chat = Chat("accounts/fireworks/models/llama-v2-70b-chat", os.getenv("Token"))
    while True:
        texto = input("")
        print(chat.run("", texto, [])["choices"][0]["message"])

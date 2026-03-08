"""Mistral client helper for HW3."""
from mimetypes import init

from dotenv import load_dotenv
import os
from mistralai import Mistral

load_dotenv()

PLANNER_MODEL  = "mistral-small-latest"  # content planning needs reasoning
STYLE_MODEL    = "mistral-small-latest"  # style needs quality output
CHUNKER_MODEL  = "mistral-small-latest"  # chunking needs reliable JSON output
CLASSIFIER_MODEL = "open-mistral-nemo"   # classification is simple, use cheapest
EMBED_MODEL    = "mistral-embed"         # embeddings, no change needed


class AppClient:
    """A Singleton wrapper for the Mistral client."""
    __client = None
    
    def __init__(self):
        pass
    
    def _load_client(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing MISTRAL_API_KEY environment variable. "
                "Set it before calling query_llm()."
            )
        try:
            AppClient.__client = Mistral(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Error loading Mistral client: {e}")
    
    @property
    def client(self):
        if AppClient.__client is None:
            self._load_client()
        return AppClient.__client


appClient = AppClient()

def query_planner_llm(prompt: str, temperature: float = 0.0) -> str:
    return query_text_llm(prompt, PLANNER_MODEL, temperature)

def query_chunker_llm(prompt: str, temperature: float = 0.0) -> str:
    return query_text_llm(prompt, CHUNKER_MODEL, temperature) 
    
def query_classifier_llm(prompt: str, temperature: float = 0.0) -> str:
    return query_text_llm(prompt, CLASSIFIER_MODEL, temperature)   
    
def query_style_llm(prompt: str, temperature: float = 0.0) -> str:
    return query_text_llm(prompt, STYLE_MODEL, temperature)


def query_text_llm(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """Send a prompt to Mistral and return the raw string response."""
    try:
        response = appClient.client.chat.complete(
            model=model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error querying LLM: {e}")



def embed(text: str) -> list[float]:
    try:   
        response = appClient.client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        return response.data[0].embedding
    
    except Exception as e:
        raise RuntimeError(f"Error querying LLM: {e}")

def embed_batch(texts: list[str]) -> list[list[float]]:
    try:   
        response = appClient.client.embeddings.create(
            model="mistral-embed",
            inputs=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"Error querying LLM: {e}")
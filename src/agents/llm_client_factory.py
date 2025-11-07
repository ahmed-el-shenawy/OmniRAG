# llm_client_factory.py
from typing import Optional
from langchain_openai import ChatOpenAI


class LLMClientFactory:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        
    ):
         
        self.api_key = api_key 
        if not self.api_key:
            raise RuntimeError("Groq API key must be provided (env var GROQ_API_KEY or parameter).")
        self.base_url = base_url

    def create_client(self, model: str, temperature: float = 0.0 ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model= model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature)
        return llm

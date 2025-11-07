# agentic_rag_service.py
from typing import Dict, Any
from .llm_client_factory import LLMClientFactory
from .sql_agent_factory import SQLAgentFactory
from .web_search_agent import WebSearchAgentFactory
from .supervisor_agent import SupervisorAgentFactory
from .rag_agent_factory import RagAgentFactory
from .vector_store_factory import VectorStoreFactory
from .embedding_service import EmbeddingService

from helpers.config import settings
from helpers.db_connection import SYNC_DATABASE_URL, DATABASE_URL

class AgenticRAGService:
    def __init__(self):
        self.services: Dict[str, Any] = {}

    @classmethod
    async def create(cls) -> "AgenticRAGService":
        self = cls()
        await self.init_services()
        return self

    async def init_services(self) -> None:
        # LLM client
        llm_client = LLMClientFactory(
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL
        ).create_client(model=settings.GROQ_MODEL, temperature=settings.TEMPERATURE)
        self.services["llm_client"] = llm_client

        # Embedding service
        embedding_svc = EmbeddingService(
            base_url=settings.OLLAMA_BASE_URL,
            api_key=settings.OLLAMA_API_KEY,
            model_name=settings.OLLAMA_MODEL
        )
        self.services["embedding_service"] = embedding_svc

        # Vector store (async)
        vector_store = await VectorStoreFactory(
            connection_string=DATABASE_URL,
            table_name=settings.VECTOR_TABLE,
            embedding_service=embedding_svc,
            vector_size=768
        ).create_vector_store()
        self.services["vector_store"] = vector_store

        # Agents
        rag_agent = RagAgentFactory(vector_store, llm_client).get_rag_agent()
        sql_agent = SQLAgentFactory(SYNC_DATABASE_URL, llm_client).build_agent()
        web_agent = WebSearchAgentFactory(llm_client).get_agent()

        self.services.update({
            "rag_agent": rag_agent,
            "sql_agent": sql_agent,
            "web_agent": web_agent
        })

        # Supervisor
        supervisor_prompt = """
You are a Supervisor Agent orchestrating three specialized worker agents. 
Your goal is to analyze the user query, call the appropriate agent(s), and return a concise, accurate answer in a structured format.

Agents:

1. rag_agent:
   - Retrieves information from documents like Ahmed's resume, reports, or textual knowledge.
   - Use when the user query relates to resumes, reports, documents, or unstructured text.
   - Example: "Summarize Ahmed's resume."

2. sql_agent:
   - Handles structured queries against the relational database.
   - Has full access to the schema (tables, columns, relationships).
   - Only perform read operations; never modify or delete data.
   - Limit results unless explicitly requested otherwise.
   - Example: "How many users registered last month?" or "Show me all orders by user Ahmed."

3. web_agent:
   - Searches the web for up-to-date or fallback information.
   - Use only if rag_agent and sql_agent cannot provide a sufficient answer.
   - Example: "What is the latest news about AI?"

Routing Rules:

- If the query explicitly refers to documents, resumes, or textual knowledge, call rag_agent first, regardless of whether it looks like structured data.
- If the query asks about the system database (users, orders, products), call sql_agent first.
- Only call web_agent if neither RAG nor SQL can answer.
- If the query is ambiguous (could be database or resume), call rag_agent and sql_agent in parallel and combine results.

- If the query has multiple components (e.g., database + document question), call each necessary agent separately and combine results into a single, concise final answer.
- After each agent call, assess if the answer is complete. Escalate to the next agent if needed.
- Always include the name of the agent that generated each part of the answer.
- Return output in structured JSON format:

Output Format:

{
  "final_answer": "Concise and clear answer combining all relevant agent responses.",
  "agents_used": ["rag_agent", "sql_agent", "web_agent"] 
}

Database Schema Context (for SQL routing):
- Users: id, name, password, role, created_at
- Orders: id, user_id, product_id, quantity, created_at
- Products: id, name, category, price, stock
- Documents: id, title, content, metadata, created_at

Example Workflows:

1. User Query: "How many users registered last month?"
   - Supervisor Action: Call sql_agent → retrieve result → return JSON with final_answer and agent_used.

2. User Query: "Summarize Ahmed's resume."
   - Supervisor Action: Call rag_agent → retrieve result → return JSON with final_answer and agent_used.

3. User Query: "How many users registered last month and summarize Ahmed's resume."
   - Supervisor Action: Call sql_agent for the first part, rag_agent for the second part → combine results into a single final_answer → return JSON with both agents listed in agents_used.

4. User Query: "What is the latest news about AI?"
   - Supervisor Action: Call rag_agent → if insufficient, call sql_agent → if still insufficient, call web_agent → combine results into final_answer → return JSON with all relevant agents used.

Constraints:
- Keep the final answer concise and precise.
- Include only necessary information from each agent.
- Always output in structured JSON format.
- Always include the names of all agents that contributed to the answer.
"""
        supervisor_agent = SupervisorAgentFactory(
            agents=[rag_agent, sql_agent, web_agent],
            model=llm_client,
            system_prompt=supervisor_prompt,
            name="main_supervisor",
            output_mode="last_message",
        ).build()
        self.services["supervisor_agent"] = supervisor_agent

    def get_service(self, name: str) -> Any:
        return self.services.get(name)

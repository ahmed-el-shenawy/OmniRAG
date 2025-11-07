# sql_agent_factory.py
from typing import Any
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

class SQLAgentFactory:
    def __init__(
        self,
        db_uri: str,
        llm_client: Any,
        top_k: int = 5,
        name: str = "sql_agent",
    ):
        self.db_uri = db_uri
        self.llm_client = llm_client
        self.top_k = top_k
        self.name = name

    def build_agent(self) -> Any:
        # Connect to the database
        db = SQLDatabase.from_uri(self.db_uri)

        # Create a SQL toolkit that wraps schema info
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm_client)

        # Extract LangChain tools (these automatically include schema)
        tools = toolkit.get_tools()

        # Optional system prompt to guide the agent behavior
        system_prompt = f"""
You are a smart SQL agent.
1. Only read data; never modify it.
2. After querying the database, do NOT return raw SQL results or how you queried the database or any information about it or it's schema.
3. Instead, summarize the results in plain, user-friendly language.
4. Highlight the most relevant fields .
5. Always limit results to top {self.top_k} rows unless the user asks otherwise.
"""
        # Create the agent
        agent = create_agent(
            model=self.llm_client,
            tools=tools,
            system_prompt=system_prompt,
            name=self.name
        )
        return agent

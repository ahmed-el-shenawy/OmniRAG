# rag_agent_factory.py
from typing import Any, Tuple, List
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document

class RagAgentFactory:
    def __init__(
        self,
        vector_store: Any,
        llm_client: Any,
        system_prompt: str = None,
        name: str = "rag_agent",
        k: int = 3,  # number of documents to retrieve
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            "You are a document retrieval agent. Use the context from retrieved documents "
            "to answer user questions accurately and shortly."
        )
        self.name = name
        self.k = k

    def _get_tool(self):
        @tool(response_format="content_and_artifact", description="Retrieve relevant documents")
        def retrieve_context(query: str) -> Tuple[str, List[Document]]:
            """
            Retrieve relevant documents from the vector store.
            Returns serialized content + raw documents as artifact.
            """
            retrieved_docs = self.vector_store.similarity_search(query, k=self.k)

            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )

            return serialized, retrieved_docs

        return retrieve_context

    def get_rag_agent(self) -> Any:
        """
        Build the RAG agent.
        """
        tool_fn = self._get_tool()
        agent = create_agent(
            self.llm_client,
            tools=[tool_fn],
            system_prompt=self.system_prompt,
            name=self.name
        )
        return agent

# vector_store_factory.py

from typing import Any
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.embeddings import Embeddings  # or your embedding service interface
import asyncio

class VectorStoreFactory:
    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_service: Embeddings,
        schema_name: str = "public",
        vector_size: int = None,   # you’ll need to pass vector dimension if initializing fresh
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_service = embedding_service
        self.schema_name = schema_name
        self.vector_size = vector_size

    async def create_vector_store(self) -> PGVectorStore:
        # 1. Create the PGEngine (async engine)
        pg_engine = PGEngine.from_connection_string(url=self.connection_string)

        # 2. Initialize the table if needed (creates id_column 'langchain_id' etc.)
        if self.vector_size is None:
            raise ValueError("vector_size must be provided when initializing a new vector store table.")
        await pg_engine.ainit_vectorstore_table(
            table_name=self.table_name,
            vector_size=self.vector_size,
            schema_name=self.schema_name
        )

        # 3. Create the vector store object
        vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=self.table_name,
            embedding_service=self.embedding_service,
            schema_name=self.schema_name
            # You can also pass id_column, content_column etc if you customized them
        )

        return vector_store

    def create_vector_store_sync(self) -> PGVectorStore:
        """
        If you really need a synchronous version, you can use create_sync,
        but your engine needs to support sync mode, and table must already exist.
        """
        # Create sync engine from string (v2 supports asyncpg or sync)
        engine = PGEngine.from_connection_string(url=self.connection_string)
        vector_store = PGVectorStore.create_sync(
            engine=engine,
            table_name=self.table_name,
            embedding_service=self.embedding_service,
            schema_name=self.schema_name
        )
        return vector_store

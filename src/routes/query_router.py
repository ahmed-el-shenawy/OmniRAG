from fastapi import APIRouter, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any
from helpers.db_connection import get_db
from helpers.handle_exceptions import handle_exceptions
from routes.schemes.query import QueryRequest
import logging

query_router = APIRouter(prefix="/query")

logger = logging.getLogger("query_router")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s"
)

@query_router.post("")
@handle_exceptions
async def answer_question(
    request: Request,
    data: QueryRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    supervisor_agent = request.app.state.supervisor_agent

    payload = {
        "messages": [
            {"role": "system", "content": "You are the user‑facing supervisor agent."},
            {"role": "user", "content": data.query}
        ]
    }

    # Invoke supervisor
    result = supervisor_agent.invoke(payload)

    # Extract agent traces
    agent_traces = getattr(result, "agent_traces", None) or result.get("agent_traces", [])

    # Extract final answer (handle both dict and AIMessage formats)
    final_answer = getattr(result, "content", None) or result.get("final_answer", None)
    if not final_answer:
        # fallback if result.messages exists
        messages = getattr(result, "messages", None) or result.get("messages", [])
        if messages:
            final_answer = getattr(messages[-1], "content", None) or messages[-1].get("content")

    # Log minimal info per agent (optional, you already log in Supervisor)
    for trace in agent_traces:
        agent_name = trace.get("agent_name")
        query_part = trace.get("arguments")
        response = trace.get("response")
        logger.info(
            f"[AGENT TRACE] Agent: {agent_name}, "
            f"Query type: {type(query_part).__name__}, "
            f"Response type: {type(response).__name__}"
        )

    # Build list of agents used
    agents_used = [trace.get("agent_name") for trace in agent_traces]

    return {
        "success": True,
        "message": None,
        "data": {
            "final_answer": final_answer,
            "agents_used": agents_used
        }
    }

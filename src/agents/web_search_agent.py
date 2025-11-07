from langchain.agents import create_agent
from langchain.tools import tool
from typing import Any, Tuple

class WebSearchAgentFactory:
    def __init__(self, llm_client: Any, system_prompt: str = None, name: str = "web_agent"):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            "You have access to a web search tool that retrieves up‑to‑date "
            "information from the internet. Use it to answer queries that need current data."
        )
        self.name = name

    def _get_tool(self):
        @tool(response_format="content_and_artifact", description="Search the web for info")
        def web_search(query: str) -> Tuple[str, Any]:
            # Use LLM client properly: pass 'input' for invoke
            result = self.llm_client.invoke(
                input=f"Perform a web search for: {query}"
            )

            content = result.output_text if hasattr(result, "output_text") else str(result)
            artifact = getattr(result, "additional_kwargs", {}).get("tool_calls", [])
            return content, artifact

        return web_search

    def get_agent(self) -> Any:
        tool_fn = self._get_tool()
        agent = create_agent(
            self.llm_client,
            tools=[tool_fn],
            system_prompt=self.system_prompt,
            name=self.name
        )
        return agent

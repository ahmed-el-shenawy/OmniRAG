# supervisor_agent.py

from typing import Any, List
from langgraph_supervisor import create_supervisor

class SupervisorAgentFactory:
    def __init__(
        self,
        agents: List[Any],
        model: Any,
        system_prompt: str,
        output_mode: str = "last_message",
        name: str = "supervisor"
    ):
        self.agents = agents
        self.model = model
        self.system_prompt = system_prompt
        self.output_mode = output_mode
        self.name = name

    def build(self) -> Any:
        supervisor = create_supervisor(
            agents=self.agents,
            model=self.model,
            prompt=self.system_prompt,
            output_mode=self.output_mode,
            supervisor_name=self.name
        ).compile(name=self.name)
        return supervisor

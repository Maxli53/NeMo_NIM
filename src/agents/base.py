from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import asyncio
from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    agent_name: str
    content: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    novelty_score: Optional[float] = None
    feasibility_score: Optional[float] = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class BaseAgent(ABC):
    def __init__(self, name: str, domain: str = "", temperature: float = 0.7):
        self.name = name
        self.domain = domain
        self.temperature = temperature
        self.message_history: List[AgentMessage] = []

    @abstractmethod
    async def respond(
        self,
        queue: asyncio.Queue,
        task: str,
        history: str,
        session_log: List[Dict[str, Any]]
    ) -> None:
        pass

    def parse_votes(self, content: str) -> Tuple[Optional[float], Optional[float]]:
        import re
        novelty_match = re.search(r"Novelty[:\s]+(\d+(?:\.\d+)?)/10", content, re.IGNORECASE)
        feasibility_match = re.search(r"Feasibility[:\s]+(\d+(?:\.\d+)?)/10", content, re.IGNORECASE)

        novelty = float(novelty_match.group(1)) if novelty_match else None
        feasibility = float(feasibility_match.group(1)) if feasibility_match else None

        return novelty, feasibility

    def format_message(
        self,
        content: str,
        citations: List[Dict[str, str]] = None
    ) -> AgentMessage:
        novelty, feasibility = self.parse_votes(content)
        message = AgentMessage(
            agent_name=self.name,
            content=content,
            citations=citations or [],
            novelty_score=novelty,
            feasibility_score=feasibility
        )
        self.message_history.append(message)
        return message

    async def send_message(
        self,
        queue: asyncio.Queue,
        content: str,
        citations: List[Dict[str, str]] = None
    ) -> None:
        message = self.format_message(content, citations)
        await queue.put(message)

    def get_history_summary(self) -> str:
        if not self.message_history:
            return ""
        return "\n\n".join([
            f"{msg.agent_name}: {msg.content}"
            for msg in self.message_history
        ])
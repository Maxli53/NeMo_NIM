from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import asyncio
from src.agents.base import BaseAgent, AgentMessage
from src.agents.consensus import ConsensusAgent


@dataclass
class DiscussionSession:
    task: str
    agents: List[BaseAgent]
    consensus_agent: ConsensusAgent
    max_rounds: int = 10
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    session_log: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: AgentMessage, citations: List[Dict[str, str]] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": message.agent_name,
            "message": message.content,
            "citations": citations or [],
            "novelty_score": message.novelty_score,
            "feasibility_score": message.feasibility_score
        }
        self.session_log.append(entry)

    def get_conversation_history(self) -> str:
        history = []
        for entry in self.session_log:
            history.append(f"{entry['agent']}: {entry['message']}")
        return "\n\n".join(history)

    def get_statistics(self) -> Dict[str, Any]:
        total_messages = len(self.session_log)
        expert_messages = [e for e in self.session_log if e['agent'] != self.consensus_agent.name]
        consensus_messages = [e for e in self.session_log if e['agent'] == self.consensus_agent.name]

        novelty_scores = [e['novelty_score'] for e in self.session_log if e.get('novelty_score')]
        feasibility_scores = [e['feasibility_score'] for e in self.session_log if e.get('feasibility_score')]

        total_citations = sum(len(e.get('citations', [])) for e in self.session_log)

        return {
            "session_id": self.session_id,
            "task": self.task,
            "total_messages": total_messages,
            "expert_messages": len(expert_messages),
            "consensus_messages": len(consensus_messages),
            "total_citations": total_citations,
            "avg_novelty": sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0,
            "avg_feasibility": sum(feasibility_scores) / len(feasibility_scores) if feasibility_scores else 0,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }

    def export_session(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = f"session_{self.session_id}.json"

        export_data = {
            "session_id": self.session_id,
            "task": self.task,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "statistics": self.get_statistics(),
            "session_log": self.session_log,
            "metadata": self.metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return filepath

    def mark_complete(self):
        self.end_time = datetime.now()


class SessionManager:
    def __init__(self):
        self.sessions: List[DiscussionSession] = []
        self.active_session: Optional[DiscussionSession] = None

    def create_session(
        self,
        task: str,
        agents: List[BaseAgent],
        consensus_agent: ConsensusAgent,
        max_rounds: int = 10
    ) -> DiscussionSession:
        session = DiscussionSession(
            task=task,
            agents=agents,
            consensus_agent=consensus_agent,
            max_rounds=max_rounds
        )
        self.sessions.append(session)
        self.active_session = session
        return session

    def get_session(self, session_id: str) -> Optional[DiscussionSession]:
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "task": s.task[:50] + "..." if len(s.task) > 50 else s.task,
                "start_time": s.start_time.isoformat(),
                "completed": s.end_time is not None,
                "message_count": len(s.session_log)
            }
            for s in self.sessions
        ]

    def export_all_sessions(self, directory: str = "sessions"):
        import os
        os.makedirs(directory, exist_ok=True)
        for session in self.sessions:
            filepath = os.path.join(directory, f"session_{session.session_id}.json")
            session.export_session(filepath)
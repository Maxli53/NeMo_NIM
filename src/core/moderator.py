import asyncio
from typing import List, Dict, Any, Optional
from src.agents.base import BaseAgent, AgentMessage
from src.agents.consensus import ConsensusAgent


class DiscussionModerator:
    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self.conversation_history: List[AgentMessage] = []

    def format_history(self) -> str:
        if not self.conversation_history:
            return ""
        return "\n\n".join([
            f"{msg.agent_name}: {msg.content}"
            for msg in self.conversation_history
        ])

    async def moderate_discussion(
        self,
        agents: List[BaseAgent],
        consensus_agent: ConsensusAgent,
        task: str,
        queue: asyncio.Queue,
        session_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        round_count = 0
        consensus_reached = False

        try:
            if agents:
                await agents[0].respond(queue, task, "", session_log)

            while round_count < self.max_rounds:
                round_count += 1

                message = await asyncio.wait_for(queue.get(), timeout=60.0)
                self.conversation_history.append(message)

                if message.agent_name == consensus_agent.name:
                    consensus_entry = next(
                        (e for e in reversed(session_log)
                         if e.get("agent") == consensus_agent.name),
                        {}
                    )
                    consensus_reached = consensus_entry.get("consensus_reached", False)

                    if consensus_reached or "CONSENSUS:" in message.content:
                        break

                    history_text = self.format_history()
                    agent_tasks = [
                        agent.respond(queue, task, history_text, session_log)
                        for agent in agents
                    ]
                    await asyncio.gather(*agent_tasks)

                else:
                    history_text = self.format_history()
                    await consensus_agent.respond(queue, task, history_text, session_log)

        except asyncio.TimeoutError:
            print(f"Discussion timed out after round {round_count}")
        except Exception as e:
            print(f"Error in moderation: {e}")

        results = self.compile_results(consensus_reached, round_count, session_log)
        return results

    def compile_results(
        self,
        consensus_reached: bool,
        round_count: int,
        session_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        novelty_scores = []
        feasibility_scores = []

        for msg in self.conversation_history:
            if msg.novelty_score is not None:
                novelty_scores.append(msg.novelty_score)
            if msg.feasibility_score is not None:
                feasibility_scores.append(msg.feasibility_score)

        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        avg_feasibility = sum(feasibility_scores) / len(feasibility_scores) if feasibility_scores else 0

        return {
            "consensus_reached": consensus_reached,
            "total_rounds": round_count,
            "total_messages": len(self.conversation_history),
            "avg_novelty": avg_novelty,
            "avg_feasibility": avg_feasibility,
            "conversation_history": self.conversation_history,
            "session_log": session_log
        }
import asyncio
from typing import List, Dict, Any, Tuple
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from src.agents.base import BaseAgent, AgentMessage
from src.config import config


class ConsensusAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ConsensusAgent",
        model: str = None,
        temperature: float = 0.2,
        client: Any = None,
        novelty_threshold: float = 7.0,
        feasibility_threshold: float = 6.0
    ):
        super().__init__(name, "Cross-disciplinary Synthesis", temperature)
        self.model = model or config.model_config.model
        self.client = client
        self.provider = config.model_config.provider
        self.novelty_threshold = novelty_threshold
        self.feasibility_threshold = feasibility_threshold

    def calculate_scores(self, session_log: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        novelty_scores = []
        feasibility_scores = []

        for entry in session_log:
            message = entry.get("message", "")
            novelty, feasibility = self.parse_votes(message)
            if novelty is not None:
                novelty_scores.append(novelty)
            if feasibility is not None:
                feasibility_scores.append(feasibility)

        return novelty_scores, feasibility_scores

    def evaluate_consensus(
        self,
        novelty_scores: List[float],
        feasibility_scores: List[float]
    ) -> Tuple[bool, float, float]:
        if not novelty_scores or not feasibility_scores:
            return False, 0.0, 0.0

        avg_novelty = sum(novelty_scores) / len(novelty_scores)
        avg_feasibility = sum(feasibility_scores) / len(feasibility_scores)

        consensus_reached = (
            avg_novelty >= self.novelty_threshold and
            avg_feasibility >= self.feasibility_threshold
        )

        return consensus_reached, avg_novelty, avg_feasibility

    async def generate_consensus_response(
        self,
        task: str,
        history: str,
        consensus_reached: bool,
        avg_novelty: float,
        avg_feasibility: float,
        session_log: List[Dict[str, Any]]
    ) -> str:
        if consensus_reached:
            system_prompt = """You are the ConsensusAgent, responsible for synthesizing interdisciplinary discussions.
Generate a comprehensive consensus summary that:
1. Highlights the key innovation proposed
2. Explains how different domains contributed
3. Lists concrete implementation steps
4. Addresses potential challenges
5. Provides clear next steps"""

            user_prompt = f"""Task: {task}

Discussion history:
{history}

Average scores - Novelty: {avg_novelty:.1f}/10, Feasibility: {avg_feasibility:.1f}/10

CONSENSUS REACHED! Please provide:
- Final Innovation Summary (2-3 sentences)
- Key Interdisciplinary Contributions (bullet points)
- Implementation Roadmap (numbered steps)
- Risk Mitigation Strategies
- Recommended Next Actions"""

        else:
            system_prompt = """You are the ConsensusAgent.
Identify gaps in the discussion and guide experts toward consensus."""

            user_prompt = f"""Task: {task}

Discussion history:
{history}

Current scores - Novelty: {avg_novelty:.1f}/10, Feasibility: {avg_feasibility:.1f}/10
Thresholds - Novelty: {self.novelty_threshold}/10, Feasibility: {self.feasibility_threshold}/10

Consensus NOT reached. Please:
1. Identify what's preventing consensus
2. Suggest specific areas needing improvement
3. Pose targeted questions to guide the discussion
4. Highlight promising ideas that need refinement"""

        if self.provider == "anthropic" and isinstance(self.client, AsyncAnthropic):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.model_config.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        elif self.provider == "openai":
            openai_client = AsyncOpenAI(api_key=config.openai_api_key)
            response = await openai_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content

        else:
            if consensus_reached:
                return f"""CONSENSUS ACHIEVED!

Final Innovation Summary:
The interdisciplinary team has successfully designed a comprehensive solution combining insights from all domains.

Key Contributions:
• Physics: Theoretical framework and quantum principles
• Biology: Bio-inspired algorithms and adaptation mechanisms
• AI/ML: Implementation architecture and learning systems
• Chemistry: Material considerations and molecular interactions

Implementation Roadmap:
1. Prototype development phase
2. Integration testing
3. Validation studies
4. Scaling considerations
5. Deployment planning

Average Novelty: {avg_novelty:.1f}/10
Average Feasibility: {avg_feasibility:.1f}/10"""
            else:
                return f"""CONSENSUS NOT YET REACHED

Current Status:
- Average Novelty: {avg_novelty:.1f}/10 (needs {self.novelty_threshold}/10)
- Average Feasibility: {avg_feasibility:.1f}/10 (needs {self.feasibility_threshold}/10)

Areas Needing Improvement:
• More innovative approaches required
• Better integration of domain expertise
• Clearer implementation pathways

Let's continue refining our approach."""

    async def respond(
        self,
        queue: asyncio.Queue,
        task: str,
        history: str,
        session_log: List[Dict[str, Any]]
    ) -> None:
        try:
            novelty_scores, feasibility_scores = self.calculate_scores(session_log)
            consensus_reached, avg_novelty, avg_feasibility = self.evaluate_consensus(
                novelty_scores, feasibility_scores
            )

            content = await self.generate_consensus_response(
                task, history, consensus_reached,
                avg_novelty, avg_feasibility, session_log
            )

            if consensus_reached:
                content = "CONSENSUS: " + content

            await self.send_message(queue, content, [])

            session_log.append({
                "agent": self.name,
                "message": content,
                "citations": [],
                "consensus_reached": consensus_reached,
                "avg_novelty": avg_novelty,
                "avg_feasibility": avg_feasibility
            })
        except Exception as e:
            error_msg = f"Error in consensus evaluation: {e}"
            await self.send_message(queue, error_msg, [])
            session_log.append({
                "agent": self.name,
                "message": error_msg,
                "citations": [],
                "consensus_reached": False
            })
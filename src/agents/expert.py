import asyncio
from typing import Optional, List, Dict, Any, Tuple
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from src.agents.base import BaseAgent
from src.core.vector_db import FAISSVectorDB
from src.config import config


class ExpertAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        domain: str,
        vector_db: Optional[FAISSVectorDB] = None,
        model: str = None,
        temperature: float = 0.7,
        client: Any = None
    ):
        super().__init__(name, domain, temperature)
        self.vector_db = vector_db
        self.model = model or config.model_config.model
        self.client = client
        self.provider = config.model_config.provider

    async def retrieve_material(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, str]]]:
        if not self.vector_db or self.vector_db.size == 0:
            return "", []

        embedding = await self.vector_db.embed_text_async(query)
        if embedding is None:
            return "", []

        results = self.vector_db.similarity_search(embedding, top_k)

        formatted_text = ""
        citations = []
        for i, result in enumerate(results, 1):
            formatted_text += f"\n[{i}] {result['text']}\n"
            citations.append({
                "source": result["source"],
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            })

        return formatted_text, citations

    async def generate_response(
        self,
        task: str,
        history: str,
        material: str
    ) -> str:
        system_prompt = f"""You are {self.name}, a world-class expert in {self.domain}.

Your role in this interdisciplinary discussion:
1. Bring domain-specific insights and expertise
2. Build on previous contributions from other experts
3. Propose innovative connections between disciplines
4. Evaluate ideas critically but constructively

Reference material (if available):
{material if material else 'No specific references available. Use your domain expertise and reasoning.'}

IMPORTANT: End your response with:
Novelty: X/10
Feasibility: Y/10"""

        user_prompt = f"""Task: {task}

Previous discussion:
{history if history else "You are starting the discussion."}

Please provide your expert perspective, considering both your domain knowledge and potential interdisciplinary connections."""

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
            return f"""{self.name} ({self.domain}):

Based on my expertise in {self.domain}, I believe this task requires careful consideration of several factors.

Without access to specific references at this moment, I would suggest exploring interdisciplinary approaches that combine {self.domain} with other fields represented here.

The key challenges from my domain perspective include technical feasibility and theoretical foundations.

I look forward to collaborating with other experts to develop a comprehensive solution.

Novelty: 6/10
Feasibility: 7/10"""

    async def respond(
        self,
        queue: asyncio.Queue,
        task: str,
        history: str,
        session_log: List[Dict[str, Any]]
    ) -> None:
        try:
            material_text, citations = await self.retrieve_material(history + " " + task)
            content = await self.generate_response(task, history, material_text)
            await self.send_message(queue, content, citations)

            session_log.append({
                "agent": self.name,
                "message": content,
                "citations": citations
            })
        except Exception as e:
            error_msg = f"I encountered an error: {e}. Proceeding with general expertise.\n\nNovelty: 5/10\nFeasibility: 5/10"
            await self.send_message(queue, error_msg, [])
            session_log.append({
                "agent": self.name,
                "message": error_msg,
                "citations": []
            })
"""
Model Manager for GPT-OSS 20B MoE and other models
Handles local inference with MoE routing and optimization
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from src.config import config, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1


class ModelManager:
    """Manages model loading and inference for local and API models"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.provider = config.model_config.provider
        self.model_name = config.model_config.model

        if config.is_local_mode():
            self._load_local_model()

    def _load_local_model(self):
        """Load GPT-OSS 20B MoE or other local models"""
        logger.info(f"Loading local model: {self.model_name}")

        try:
            # Configure quantization if specified
            quantization_config = None
            if config.model_config.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            elif config.model_config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with MoE support
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=config.model_config.device,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            # Set device
            if config.model_config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.model_config.device

            logger.info(f"Model loaded successfully on {self.device}")

            # Log MoE configuration if available
            if hasattr(self.model.config, "num_experts"):
                logger.info(f"MoE Configuration: {self.model.config.num_experts} experts, "
                           f"{getattr(self.model.config, 'num_experts_per_tok', 'N/A')} active per token")

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using the loaded model"""

        if not config.is_local_mode():
            return self._generate_api(prompt, params, system_prompt)

        if params is None:
            params = GenerationParams()

        # Format prompt with system message if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Tokenize input
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        if self.device == "cuda":
            inputs = inputs.to("cuda")

        # Generate with MoE routing
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                do_sample=params.do_sample,
                repetition_penalty=params.repetition_penalty,
                num_return_sequences=params.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove input prompt from output
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()

        return generated_text

    def _generate_api(
        self,
        prompt: str,
        params: Optional[GenerationParams],
        system_prompt: Optional[str]
    ) -> str:
        """Generate using API (Anthropic/OpenAI) - fallback option"""
        # This would contain API generation logic
        # For now, return placeholder
        return f"[API Mode] Response to: {prompt[:50]}..."

    async def generate_async(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Async wrapper for generation"""
        import asyncio
        return await asyncio.to_thread(
            self.generate, prompt, params, system_prompt
        )

    def batch_generate(
        self,
        prompts: List[str],
        params: Optional[GenerationParams] = None
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, params)
            responses.append(response)
        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "device": self.device,
            "quantization": config.model_config.quantization
        }

        if config.is_local_mode() and self.model:
            info.update({
                "model_size": sum(p.numel() for p in self.model.parameters()),
                "dtype": str(next(self.model.parameters()).dtype),
            })

            # Add MoE info if available
            if hasattr(self.model.config, "num_experts"):
                info["moe_experts"] = self.model.config.num_experts
                info["moe_active_experts"] = getattr(
                    self.model.config, "num_experts_per_tok", "N/A"
                )

        return info


class ModelPool:
    """Manages multiple models for different purposes"""

    def __init__(self):
        self.models = {}
        self.primary_model = None

    def load_model(self, name: str, model_path: str) -> ModelManager:
        """Load a model into the pool"""
        manager = ModelManager()
        self.models[name] = manager
        if self.primary_model is None:
            self.primary_model = manager
        return manager

    def get_model(self, name: Optional[str] = None) -> ModelManager:
        """Get a specific model or the primary model"""
        if name:
            return self.models.get(name)
        return self.primary_model


# Global model manager instance
model_manager = ModelManager()
"""
LLM Interface supporting CBORG API (and legacy Claude/OpenAI)
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import json

from pathlib import Path

# Optional dotenv support
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Optional imports for legacy support
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Load .env from project root
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists() and load_dotenv:
    load_dotenv(env_path)
elif load_dotenv:
    # Fallback to default load_dotenv behavior
    load_dotenv()


class LLMProvider(str, Enum):
    CBORG = "cborg"
    CLAUDE = "claude"
    OPENAI = "openai"


class BaseLLM(ABC):
    """Base class for LLM implementations"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass

    @abstractmethod
    async def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response"""
        pass


class CborgLLM(BaseLLM):
    """CBORG API implementation (OpenAI-compatible)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("CBORG_API_KEY")
        self.base_url = base_url or os.getenv(
            "CBORG_BASE_URL", "https://api.cborg.lbl.gov"
        )
        self.model = model or os.getenv("CBORG_MODEL", "google/gemini-flash-lite")

        if not self.api_key:
            raise ValueError("CBORG_API_KEY not found in environment variables")

        # Remove quotes if present
        self.api_key = self.api_key.strip("\"'")
        self.base_url = self.base_url.strip("\"'")
        self.model = self.model.strip("\"'")

        # Create OpenAI client configured for CBORG
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using CBORG API"""

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    async def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response"""

        # Add schema instructions to prompt
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Important: Return ONLY valid JSON, no additional text or formatting.
"""

        response = await self.generate(structured_prompt, **kwargs)

        # Parse JSON response
        try:
            # Clean the response - sometimes models add markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse structured response: {response}")


class ClaudeLLM(BaseLLM):
    """Claude (Anthropic) implementation"""

    def __init__(self, api_key: Optional[str] = None):
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Install with: pixi add anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-20241022"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Claude"""

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    async def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response"""

        # Add schema instructions to prompt
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""

        response = await self.generate(structured_prompt, **kwargs)

        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            raise ValueError("Could not parse structured response")


class OpenAILLM(BaseLLM):
    """OpenAI implementation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4-turbo-preview"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI"""

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    async def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response using function calling"""

        # Convert schema to OpenAI function format
        function = {
            "name": "respond_with_structure",
            "description": "Respond with structured data",
            "parameters": {
                "type": "object",
                "properties": schema,
                "required": list(schema.keys()),
            },
        }

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            functions=[function],
            function_call={"name": "respond_with_structure"},
        )

        return json.loads(response.choices[0].message.function_call.arguments)


class LLMInterface:
    """Unified interface for LLM providers"""

    def __init__(self, provider: LLMProvider = LLMProvider.CBORG):
        self.provider = provider

        if provider == LLMProvider.CBORG:
            self.llm = CborgLLM()
        elif provider == LLMProvider.CLAUDE:
            self.llm = ClaudeLLM()
        elif provider == LLMProvider.OPENAI:
            self.llm = OpenAILLM()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response"""
        return await self.llm.generate(prompt, **kwargs)

    async def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response"""
        return await self.llm.generate_structured(prompt, schema, **kwargs)

    def switch_provider(self, provider: LLMProvider):
        """Switch LLM provider"""
        self.__init__(provider)

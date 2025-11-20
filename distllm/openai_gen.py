from __future__ import annotations

import os
import openai
from pydantic import Field
import logging

from distllm.utils import BaseConfig

logger = logging.getLogger(__name__)


class OpenAIGeneratorConfig(BaseConfig):
    """Configuration for the OpenAI generator using OpenAI client."""

    model: str = Field(
        default_factory=lambda: os.getenv('MODEL', 'gpt-4o'),
        description='The model name for OpenAI proxy.',
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'OPENAI_BASE_URL',
            'https://api.openai.com/v1',
        ),
        description='The base URL for the OpenAI proxy server.',
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            'OPENAI_API_KEY',
            None,
        ),
        description='The API key for OpenAI services. Required',
    )
    temperature: float = Field(
        1.0,
        description='Freeze off the temperature to keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> OpenAIGenerator:
        """Get the OpenAI generator."""
        generator = OpenAIGenerator(
            config=self,
        )
        return generator


class OpenAIGenerator:
    """A generator that calls the Argo proxy using OpenAI client."""

    def __init__(self, config: OpenAIGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        logger.warning(f"IGNORING TEMPERATURE")
        assert config.api_key, 'Chat system requires OPENAI_API_KEY'

        # Initialize OpenAI client with Argo proxy settings
        self.client = openai.OpenAI(
            api_key=config.api_key,
        )

        try:
            self.client.models.list()
        except Exception as e:
            logger.exception('Connecting to OpenAI services failed')

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt to the Argo proxy and return the completion."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # temperature=temp_to_use,
                max_completion_tokens=tokens_to_use,
            )
            result = response.choices[0].message.content
        except Exception as e:
            logger.exception('Error calling OpenAI servers')
            result = f'Error: {e!s}'

        return result


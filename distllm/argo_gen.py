from __future__ import annotations

import os
import openai
from pydantic import Field
import logging

from distllm.utils import BaseConfig

logger = logging.getLogger(__name__)

class ArgoGeneratorConfig(BaseConfig):
    """Configuration for the Argo generator using OpenAI client."""

    model: str = Field(
        default_factory=lambda: os.getenv('MODEL', 'argo:gpt-4o'),
        description='The model name for Argo proxy.',
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'BASE_URL',
            'http://localhost:56267',
        ),
        description='The base URL for the Argo proxy server.',
    )
    api_key: str = Field(
        'whatever+random',
        description='The API key for Argo proxy (can be any string).',
    )
    temperature: float = Field(
        0.0,
        description='Freeze off the temperature to keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> ArgoGenerator:
        """Get the Argo generator."""
        generator = ArgoGenerator(
            config=self,
        )
        return generator


class ArgoGenerator:
    """A generator that calls the Argo proxy using OpenAI client."""

    def __init__(self, config: ArgoGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Initialize OpenAI client with Argo proxy settings
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=f'{config.base_url}/v1',
        )

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
                temperature=temp_to_use,
                max_tokens=tokens_to_use,
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f'Error calling Argo proxy: {e}')
            result = f'Error: {e!s}'

        return result

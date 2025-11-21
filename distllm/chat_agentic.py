from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openai
import requests
from academy.agent import Agent, action
from academy.exception import ForbiddenError
from academy.exchange.cloud import HttpExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from dotenv import load_dotenv
from pydantic import Field

from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig
from distllm.rag.search import Retriever, RetrieverConfig
from distllm.utils import BaseConfig
from distllm.distllm.chat_openai import ChatAppConfig


EXCHANGE_ADDRESS = 'https://exchange.proxystore.dev'


class HPRAGAgent(Agent):

    def __init__(self, config: ChatAppConfig):
        self.config = config

    async def agent_on_startup(self) -> None:


    @action
    async def machine_info(self) -> dict[str, t.Any]:

        import platform
        import sys
        import dill

        return {
            'platinfo': platform.uname,
            'sys_info': str(sys.version_info),
            'dill': dill.__version__,
        }




async def main(config: ChatAppConfig) -> None:
    init_logging('INFO')

    with open('sharing.json') as f:
        sharing_info = json.load(f)

    sharing_group_id = sharing_info['academy_sharing_group']

    factory = HttpExchangeFactory(url=EXCHANGE_ADDRESS, auth_method='globus')
    async with await Manager.from_exchange_factory(
        factory=factory, executors=ThreadPoolExecutor()
    ) as manager:
        console = await factory.console()
        handle = await manager.launch(HPRAGAgent(config=config))
        await handle.ping()

        try:
            await console.share_mailbox(manager.user_id, sharing_group_id)
            await console.share_mailbox(handle.agent_id, sharing_group_id)
        except ForbiddenError:
            logging.warning('Failed to share mailbox. Check group membership')
            raise

        result = await handle.machine_info()
        print('Got result:', result)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = ChatAppConfig.from_yaml(args.config)

    # Start the interactive chat
    # chat_with_model(config)

    asyncio.run(main(config))


if __name__ == '__main__':

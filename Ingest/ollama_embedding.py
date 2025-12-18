from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_community.embeddings import OllamaEmbeddings

from langflow.base.models.model import LCModelComponent
from langflow.base.models.ollama_constants import URL_LIST
from langflow.field_typing import Embeddings
from langflow.io import DropdownInput, MessageTextInput, Output

HTTP_STATUS_OK = 200


class OllamaEmbeddingsComponent(LCModelComponent):
    display_name: str = "Ollama Embeddings"
    description: str = "Generate embeddings using Ollama models."
    documentation = "https://python.langchain.com/docs/integrations/text_embedding/ollama"
    icon = "Ollama"
    name = "OllamaEmbeddings"

    inputs = [
        DropdownInput(
            name="model_name",
            display_name="Ollama Model",
            value="",
            options=[],
            real_time_refresh=True,
            refresh_button=True,
            combobox=True,
            required=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Ollama Base URL",
            value="",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Embeddings",
            name="embeddings",
            method="build_embeddings",
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        try:
            return OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url,
            )
        except Exception as e:
            raise ValueError(
                "Unable to connect to the Ollama API. "
                "Verify the base URL and ensure the model is pulled."
            ) from e

    async def update_build_config(
        self,
        build_config: dict,
        field_value: Any,
        field_name: str | None = None,
    ):
        """
        FIXED behavior:
        - Base URL is ONLY validated when base_url changes
        - Refreshing model list never wipes base_url
        - Model list refreshes when base_url or model_name changes
        """

        # ✅ Only validate URL when base_url itself changes
        if field_name == "base_url":
            if not await self.is_valid_ollama_url(field_value):
                valid_url = ""
                for url in URL_LIST:
                    if await self.is_valid_ollama_url(url):
                        valid_url = url
                        break
                build_config["base_url"]["value"] = valid_url

        # ✅ Refresh models when base_url or model_name changes
        if field_name in {"base_url", "model_name"}:
            base_url = (
                build_config.get("base_url", {}).get("value")
                or self.base_url
            )

            if base_url and await self.is_valid_ollama_url(base_url):
                build_config["model_name"]["options"] = await self.get_models(base_url)
            else:
                build_config["model_name"]["options"] = []

        return build_config

    async def get_models(self, base_url_value: str) -> list[str]:
        """Get ALL model names from Ollama."""
        try:
            url = urljoin(base_url_value, "/api/tags")
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            return sorted(
                model["name"]
                for model in data.get("models", [])
                if "name" in model
            )

        except Exception as e:
            raise ValueError("Could not get model names from Ollama.") from e

    async def is_valid_ollama_url(self, url: str) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/api/tags")
                return response.status_code == HTTP_STATUS_OK
        except httpx.RequestError:
            return False
import time
from http import HTTPStatus
from typing import Any

import aiohttp
from app.core.logger import get_logger
from app.exceptions.exceptions import (
    ProviderAPIException,
    BaseInvalidRequestException,
)

from .base import ProviderAdapter

logger = get_logger(name="ideogram_adapter")


class IdeogramAdapter(ProviderAdapter):
    """Adapter for Ideogram API"""

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        config: dict[str, Any] | None = None,
    ):
        self._provider_name = provider_name
        self._base_url = base_url

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def get_model_id(self, payload: dict[str, Any]) -> str:
        """Get the model ID from the payload"""
        if "model" in payload:
            return payload["model"]
        else:
            logger.error(f"Model ID not found in payload for {self.provider_name}")
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Model ID not found in payload"),
            )

    async def list_models(
        self,
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> list[str]:
        """List all models supported by Ideogram"""
        # Check cache first
        base_url = base_url or self._base_url
        cached_models = self.get_cached_models(api_key, base_url)
        if cached_models is not None:
            return cached_models

        # Ideogram models (based on their documentation)
        models = [
            "ideogram-v3",
            "ideogram-v2",
            "ideogram-v1-turbo",
            "ideogram-v1",
        ]

        # Cache the results
        self.cache_models(api_key, base_url, models)
        return models

    async def process_completion(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Ideogram doesn't support text completion"""
        raise NotImplementedError(
            f"Ideogram doesn't support text completion endpoint: {endpoint}"
        )

    async def process_embeddings(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Ideogram doesn't support embeddings"""
        raise NotImplementedError(
            f"Ideogram doesn't support embeddings endpoint: {endpoint}"
        )

    async def process_image_generation(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> Any:
        """Process an image generation request using Ideogram API"""
        prompt = payload.get("prompt")
        
        if not prompt:
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Prompt is required for image generation"),
            )

        # Map OpenAI-style parameters to Ideogram format
        ideogram_payload = {
            "prompt": prompt,
            "rendering_speed": self._map_quality_to_speed(payload.get("quality", "standard")),
        }

        # Add model if specified
        model = payload.get("model", "ideogram-v3")
        if model and model != "ideogram-v3":
            ideogram_payload["model"] = model

        # Add aspect ratio if size is specified
        size = payload.get("size", "1024x1024")
        aspect_ratio = self._convert_size_to_aspect_ratio(size)
        if aspect_ratio:
            ideogram_payload["aspect_ratio"] = aspect_ratio

        # Add style parameters
        style = payload.get("style")
        if style:
            ideogram_payload["style_type"] = self._map_style(style)

        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/v1/ideogram-v3/generate"

        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=headers, json=ideogram_payload) as response,
        ):
            if response.status != HTTPStatus.OK:
                error_text = await response.text()
                logger.error(
                    f"Image Generation API error for {self.provider_name}: {error_text}"
                )
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=response.status,
                    error_message=error_text,
                )

            result = await response.json()
            
            # Convert Ideogram response to OpenAI-compatible format
            if "data" in result and len(result["data"]) > 0:
                return {
                    "created": int(time.time()),
                    "data": [
                        {
                            "url": item["url"],
                            "revised_prompt": prompt,
                        }
                        for item in result["data"]
                    ],
                }
            else:
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=500,
                    error_message="No image data returned from Ideogram API",
                )

    def _map_quality_to_speed(self, quality: str) -> str:
        """Map OpenAI quality parameter to Ideogram rendering speed"""
        quality_map = {
            "standard": "TURBO",
            "hd": "STANDARD",
        }
        return quality_map.get(quality.lower(), "TURBO")

    def _convert_size_to_aspect_ratio(self, size: str) -> str | None:
        """Convert OpenAI size parameter to Ideogram aspect ratio"""
        size_map = {
            "256x256": "ASPECT_1_1",
            "512x512": "ASPECT_1_1", 
            "1024x1024": "ASPECT_1_1",
            "1792x1024": "ASPECT_16_9",
            "1024x1792": "ASPECT_9_16",
            "1536x1024": "ASPECT_3_2",
            "1024x1536": "ASPECT_2_3",
        }
        return size_map.get(size)

    def _map_style(self, style: str) -> str:
        """Map OpenAI style parameter to Ideogram style type"""
        style_map = {
            "vivid": "GENERAL",
            "natural": "REALISTIC",
        }
        return style_map.get(style.lower(), "GENERAL")
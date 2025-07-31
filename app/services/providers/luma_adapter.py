import asyncio
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

logger = get_logger(name="luma_adapter")


class LumaAdapter(ProviderAdapter):
    """Adapter for Luma AI API"""

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
        """List all models supported by Luma AI"""
        # Check cache first
        base_url = base_url or self._base_url
        cached_models = self.get_cached_models(api_key, base_url)
        if cached_models is not None:
            return cached_models

        # Luma AI models (based on their documentation)
        models = [
            "photon-1",
            "photon-flash-1",
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
        """Luma AI doesn't support text completion"""
        raise NotImplementedError(
            f"Luma AI doesn't support text completion endpoint: {endpoint}"
        )

    async def process_embeddings(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Luma AI doesn't support embeddings"""
        raise NotImplementedError(
            f"Luma AI doesn't support embeddings endpoint: {endpoint}"
        )

    async def process_image_generation(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> Any:
        """Process an image generation request using Luma AI API"""
        prompt = payload.get("prompt")
        model = payload.get("model", "photon-1")
        
        if not prompt:
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Prompt is required for image generation"),
            )

        # Map OpenAI-style parameters to Luma AI format
        luma_payload = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": self._convert_size_to_aspect_ratio(payload.get("size", "1024x1024")),
        }

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/dream-machine/v1/generations/image"

        # Submit the generation request
        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=headers, json=luma_payload) as response,
        ):
            if response.status not in [HTTPStatus.OK, HTTPStatus.CREATED]:
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
            generation_id = result.get("id")

            if not generation_id:
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=500,
                    error_message="No generation ID returned from Luma AI API",
                )

        # Poll for generation completion
        max_attempts = 60  # 60 attempts * 5 seconds = 5 minutes maximum wait time
        attempt = 0

        while attempt < max_attempts:
            # Wait before checking status
            await asyncio.sleep(5)
            
            status_url = f"{self._base_url}/dream-machine/v1/generations/{generation_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(status_url, headers=headers) as status_response:
                    if status_response.status == HTTPStatus.OK:
                        status_data = await status_response.json()
                        state = status_data.get("state")
                        
                        logger.debug(
                            f"Attempt {attempt + 1}/{max_attempts}: Generation state is '{state}'"
                        )
                        
                        if state == "completed":
                            assets = status_data.get("assets", {})
                            image_url = assets.get("image")
                            if image_url:
                                # Return OpenAI-compatible response format
                                return {
                                    "created": int(time.time()),
                                    "data": [
                                        {
                                            "url": image_url,
                                            "revised_prompt": prompt,
                                        }
                                    ],
                                }
                            else:
                                raise ProviderAPIException(
                                    provider_name=self.provider_name,
                                    error_code=500,
                                    error_message="Generation completed but no image URL found",
                                )
                        elif state == "failed":
                            failure_reason = status_data.get("failure_reason", "Unknown error")
                            raise ProviderAPIException(
                                provider_name=self.provider_name,
                                error_code=500,
                                error_message=f"Generation failed: {failure_reason}",
                            )
                        else:
                            attempt += 1
                    else:
                        error_text = await status_response.text()
                        raise ProviderAPIException(
                            provider_name=self.provider_name,
                            error_code=status_response.status,
                            error_message=f"Error checking generation status: {error_text}",
                        )

        # If we've exhausted all polling attempts
        raise ProviderAPIException(
            provider_name=self.provider_name,
            error_code=408,
            error_message="Image generation timed out. Maximum polling attempts reached.",
        )

    def _convert_size_to_aspect_ratio(self, size: str) -> str:
        """Convert OpenAI size parameter to Luma AI aspect ratio format"""
        size_map = {
            "256x256": "1:1",
            "512x512": "1:1",
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1536x1024": "3:2",
            "1024x1536": "2:3",
            "1920x1080": "16:9",
        }
        return size_map.get(size, "1:1")
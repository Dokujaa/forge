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

logger = get_logger(name="runway_adapter")


class RunwayAdapter(ProviderAdapter):
    """Adapter for Runway API"""

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
        """List all models supported by Runway"""
        # Check cache first
        base_url = base_url or self._base_url
        cached_models = self.get_cached_models(api_key, base_url)
        if cached_models is not None:
            return cached_models

        # Runway models (based on their documentation)
        models = [
            "gen4_image",
            "gen3_image",
            "gen2_image",
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
        """Runway doesn't support text completion"""
        raise NotImplementedError(
            f"Runway doesn't support text completion endpoint: {endpoint}"
        )

    async def process_embeddings(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Runway doesn't support embeddings"""
        raise NotImplementedError(
            f"Runway doesn't support embeddings endpoint: {endpoint}"
        )

    async def process_image_generation(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> Any:
        """Process an image generation request using Runway API"""
        prompt = payload.get("prompt")
        model = payload.get("model", "gen4_image")
        
        if not prompt:
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Prompt is required for image generation"),
            )

        # Map OpenAI-style parameters to Runway format
        runway_payload = {
            "model": model,
            "prompt_text": prompt,
            "ratio": self._convert_size_to_ratio(payload.get("size", "1024x1024")),
        }

        # Add seed if provided
        seed = payload.get("seed")
        if seed:
            runway_payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/v1/text_to_image"

        # Submit the generation task
        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=headers, json=runway_payload) as response,
        ):
            if response.status != HTTPStatus.CREATED and response.status != HTTPStatus.OK:
                error_text = await response.text()
                logger.error(
                    f"Image Generation API error for {self.provider_name}: {error_text}"
                )
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=response.status,
                    error_message=error_text,
                )

            task_result = await response.json()
            task_id = task_result.get("id")

            if not task_id:
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=500,
                    error_message="No task ID returned from Runway API",
                )

        # Poll for task completion
        max_attempts = 60  # 60 attempts * 2 seconds = 2 minutes maximum wait time
        attempt = 0

        while attempt < max_attempts:
            # Wait before checking status
            await asyncio.sleep(2)
            
            async with aiohttp.ClientSession() as session:
                task_url = f"{self._base_url}/v1/tasks/{task_id}"
                async with session.get(task_url, headers=headers) as status_response:
                    if status_response.status == HTTPStatus.OK:
                        task_data = await status_response.json()
                        status = task_data.get("status")
                        
                        if status == "SUCCEEDED":
                            output = task_data.get("output")
                            if output and len(output) > 0:
                                # Return OpenAI-compatible response format
                                return {
                                    "created": int(time.time()),
                                    "data": [
                                        {
                                            "url": output[0],
                                            "revised_prompt": prompt,
                                        }
                                    ],
                                }
                            else:
                                raise ProviderAPIException(
                                    provider_name=self.provider_name,
                                    error_code=500,
                                    error_message="Task succeeded but no output found",
                                )
                        elif status == "FAILED":
                            error_message = task_data.get("error", "Unknown error")
                            raise ProviderAPIException(
                                provider_name=self.provider_name,
                                error_code=500,
                                error_message=f"Generation failed: {error_message}",
                            )
                        else:
                            logger.debug(
                                f"Task not ready yet (attempt {attempt + 1}/{max_attempts}): {status}"
                            )
                            attempt += 1
                    else:
                        error_text = await status_response.text()
                        raise ProviderAPIException(
                            provider_name=self.provider_name,
                            error_code=status_response.status,
                            error_message=f"Error checking task status: {error_text}",
                        )

        # If we've exhausted all polling attempts
        raise ProviderAPIException(
            provider_name=self.provider_name,
            error_code=408,
            error_message="Image generation timed out. Maximum polling attempts reached.",
        )

    def _convert_size_to_ratio(self, size: str) -> str:
        """Convert OpenAI size parameter to Runway ratio format"""
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
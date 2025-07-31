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

logger = get_logger(name="blackforest_adapter")


class BlackForestAdapter(ProviderAdapter):
    """Adapter for Black Forest Labs (Flux) API"""

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
        """List all models supported by Black Forest Labs"""
        # Check cache first
        base_url = base_url or self._base_url
        cached_models = self.get_cached_models(api_key, base_url)
        if cached_models is not None:
            return cached_models

        # Black Forest Labs models (based on exploration script)
        models = [
            "flux-pro-1.1",
            "flux-pro",
            "flux-dev",
            "flux-schnell",
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
        """Black Forest Labs doesn't support text completion"""
        raise NotImplementedError(
            f"Black Forest Labs doesn't support text completion endpoint: {endpoint}"
        )

    async def process_embeddings(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Black Forest Labs doesn't support embeddings"""
        raise NotImplementedError(
            f"Black Forest Labs doesn't support embeddings endpoint: {endpoint}"
        )

    async def process_image_generation(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> Any:
        """Process an image generation request using Black Forest Labs API"""
        model = payload.get("model", "flux-pro-1.1")
        prompt = payload.get("prompt")
        
        if not prompt:
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Prompt is required for image generation"),
            )

        # Map OpenAI-style parameters to Black Forest Labs format
        bfl_payload = {
            "prompt": prompt,
            "width": self._parse_size_param(payload.get("size", "1024x1024"))[0],
            "height": self._parse_size_param(payload.get("size", "1024x1024"))[1],
            "prompt_upsampling": False,
            "seed": payload.get("seed", 42),
            "safety_tolerance": 2,
            "output_format": "jpeg",  # BlackForest expects 'jpeg' or 'png'
        }

        headers = {
            "x-key": api_key,
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/v1/{model}"

        # Submit the generation task
        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=headers, json=bfl_payload) as response,
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

            submission_result = await response.json()
            polling_url = submission_result.get("polling_url")

            if not polling_url:
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=500,
                    error_message="No polling URL returned from Black Forest Labs API",
                )

        # Poll for results
        max_attempts = 30  # 30 attempts * 2 seconds = 1 minute maximum wait time
        attempt = 0

        while attempt < max_attempts:
            async with aiohttp.ClientSession() as session:
                async with session.get(polling_url) as polling_response:
                    if polling_response.status == HTTPStatus.OK:
                        polling_data = await polling_response.json()
                        
                        if polling_data.get("status") == "Ready":
                            image_url = polling_data.get("result", {}).get("sample")
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
                        elif polling_data.get("status") == "Error":
                            raise ProviderAPIException(
                                provider_name=self.provider_name,
                                error_code=500,
                                error_message=f"Generation failed: {polling_data.get('error', 'Unknown error')}",
                            )
                        else:
                            logger.debug(
                                f"Task not ready yet (attempt {attempt + 1}/{max_attempts}): {polling_data.get('status')}"
                            )
                            await asyncio.sleep(2)
                            attempt += 1
                    else:
                        error_text = await polling_response.text()
                        raise ProviderAPIException(
                            provider_name=self.provider_name,
                            error_code=polling_response.status,
                            error_message=f"Error polling for results: {error_text}",
                        )

        # If we've exhausted all polling attempts
        raise ProviderAPIException(
            provider_name=self.provider_name,
            error_code=408,
            error_message="Image generation timed out. Maximum polling attempts reached.",
        )

    def _parse_size_param(self, size_param: str) -> tuple[int, int]:
        """Parse OpenAI-style size parameter (e.g., '1024x1024') to width, height tuple"""
        try:
            if 'x' in size_param:
                width, height = map(int, size_param.split('x'))
                return width, height
            else:
                # Default fallback
                return 1024, 1024
        except (ValueError, AttributeError):
            logger.warning(f"Invalid size parameter: {size_param}, using default 1024x1024")
            return 1024, 1024
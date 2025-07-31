import time
from http import HTTPStatus
from typing import Any

import aiohttp
import requests  # Added for Stability AI API calls
from app.core.logger import get_logger
from app.exceptions.exceptions import (
    ProviderAPIException,
    BaseInvalidRequestException,
)

from .base import ProviderAdapter

logger = get_logger(name="stability_adapter")


class StabilityAdapter(ProviderAdapter):
    """Adapter for Stability AI API"""

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
        """List all models supported by Stability AI"""
        # Check cache first  
        base_url = base_url or self._base_url
        cached_models = self.get_cached_models(api_key, base_url)
        if cached_models is not None:
            return cached_models

        # Stability AI models (based on their documentation)
        models = [
            "stable-image-ultra",
            "stable-image-core",
            "stable-diffusion-v1-6",
            "stable-diffusion-xl-1024-v1-0",
            "stable-diffusion-3-medium",
            "stable-diffusion-3-large",
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
        """Stability AI doesn't support text completion"""
        raise NotImplementedError(
            f"Stability AI doesn't support text completion endpoint: {endpoint}"
        )

    async def process_embeddings(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
        base_url: str | None = None,
        query_params: dict[str, Any] = None,
    ) -> Any:
        """Stability AI doesn't support embeddings"""
        raise NotImplementedError(
            f"Stability AI doesn't support embeddings endpoint: {endpoint}"
        )

    async def process_image_generation(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str,
    ) -> Any:
        """Process an image generation request using Stability AI API"""
        prompt = payload.get("prompt")
        model = payload.get("model", "stable-image-ultra")
        
        if not prompt:
            raise BaseInvalidRequestException(
                provider_name=self.provider_name,
                error=ValueError("Prompt is required for image generation"),
            )

        # Determine the API endpoint based on model
        model_endpoint = self._get_model_endpoint(model)
        
        # Prepare form data according to Stability AI documentation
        data = {
            "prompt": prompt,
        }
        
        # Map OpenAI response_format to Stability AI output_format
        response_format = payload.get("response_format", "url")
        if response_format == "url":
            # When OpenAI format is "url", default to png
            output_format = "png"
        else:
            # Use the format directly if it's jpeg/png/webp
            output_format = response_format
        data["output_format"] = output_format
        
        # Handle aspect ratio for applicable models
        if "ultra" in model.lower() or "core" in model.lower():
            # Stable Image Ultra and Core use aspect_ratio
            aspect_ratio = self._convert_size_to_aspect_ratio(payload.get("size", "1024x1024"))
            if aspect_ratio:
                data["aspect_ratio"] = aspect_ratio
        
        # Add optional parameters
        if payload.get("seed"):
            data["seed"] = str(payload["seed"])
            
        if payload.get("negative_prompt"):
            data["negative_prompt"] = payload["negative_prompt"]
            
        if payload.get("style"):
            data["style_preset"] = self._map_style(payload["style"])

        # Prepare files dict (required even when empty according to docs)
        files = {"none": ""}
        
        # Set up headers according to documentation
        headers = {
            "authorization": f"Bearer {api_key}",
            "accept": "image/*",
            # Do not set content-type - requests library will set it automatically for multipart
        }

        url = f"{self._base_url}/v2beta/{model_endpoint}"

        # Use requests-style approach as shown in Stability documentation
        try:
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(
                    f"Image Generation API error for {self.provider_name}: {error_text}"
                )
                raise ProviderAPIException(
                    provider_name=self.provider_name,
                    error_code=response.status_code,
                    error_message=error_text,
                )

            # Stability AI returns the image directly as binary data
            image_data = response.content
            
            # Convert binary data to base64 for URL format
            import base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine the MIME type based on the response format
            mime_type = f"image/{output_format}"
            
            return {
                "created": int(time.time()),
                "data": [
                    {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "revised_prompt": prompt,
                    }
                ],
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error for {self.provider_name}: {str(e)}")
            raise ProviderAPIException(
                provider_name=self.provider_name,
                error=e,
            )

    def _get_model_endpoint(self, model: str) -> str:
        """Get the API endpoint for the specified model"""
        model_endpoints = {
            "stable-image-ultra": "stable-image/generate/ultra",
            "stable-image-core": "stable-image/generate/core", 
            "stable-diffusion-v1-6": "stable-image/generate/sd3",
            "stable-diffusion-xl-1024-v1-0": "stable-image/generate/sdxl",
            "stable-diffusion-3-medium": "stable-image/generate/sd3",
            "stable-diffusion-3-large": "stable-image/generate/sd3",
        }
        return model_endpoints.get(model, "stable-image/generate/ultra")

    def _convert_size_to_aspect_ratio(self, size: str) -> str | None:
        """Convert OpenAI size parameter to Stability AI aspect ratio"""
        size_map = {
            "256x256": "1:1",
            "512x512": "1:1",
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1536x1024": "3:2",
            "1024x1536": "2:3",
        }
        return size_map.get(size)

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

    def _map_style(self, style: str) -> str:
        """Map OpenAI style parameter to Stability AI style preset"""
        style_map = {
            "vivid": "enhance",
            "natural": "photographic",
        }
        return style_map.get(style.lower(), "enhance")
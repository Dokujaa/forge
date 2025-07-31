#!/usr/bin/env python3
"""
Simple test script to verify image generation adapters work correctly
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.services.providers.blackforest_adapter import BlackForestAdapter
from app.services.providers.openai_adapter import OpenAIAdapter
from app.services.providers.stability_adapter import StabilityAdapter


async def test_openai_adapter():
    """Test OpenAI adapter"""
    print("\n" + "="*50)
    print("Testing OpenAI Adapter")
    print("="*50)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping test.")
        return False
    
    try:
        # Create adapter
        adapter = OpenAIAdapter(
            provider_name="openai",
            base_url="https://api.openai.com/v1"
        )
        
        # Test parameters
        payload = {
            "prompt": "A beautiful sunset over a calm ocean with palm trees",
            "model": "dall-e-3",
            "size": "1024x1024",
            "response_format": "url",
            "n": 1
        }
        
        print("🚀 Starting image generation...")
        print(f"Model: {payload['model']}")
        print(f"Prompt: {payload['prompt']}")
        
        # Generate image
        result = await adapter.process_image_generation(
            endpoint="images/generations",
            payload=payload,
            api_key=api_key
        )
        
        print("✅ Image generation successful!")
        print(f"Response keys: {list(result.keys())}")
        if "data" in result and len(result["data"]) > 0:
            print(f"Image URL: {result['data'][0].get('url', 'No URL')}")
            print(f"Revised prompt: {result['data'][0].get('revised_prompt', 'No revised prompt')}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing OpenAI adapter: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_blackforest_adapter():
    """Test BlackForest Labs adapter"""
    print("\n" + "="*50)
    print("Testing BlackForest Labs Adapter")
    print("="*50)
    
    # Get API key from environment
    api_key = os.getenv("BLACKFOREST_API_KEY")
    if not api_key:
        print("❌ BLACKFOREST_API_KEY not set. Skipping test.")
        return False
    
    try:
        # Create adapter
        adapter = BlackForestAdapter(
            provider_name="blackforest",
            base_url="https://api.bfl.ai"
        )
        
        # Test parameters
        payload = {
            "prompt": "A beautiful sunset over a calm ocean with palm trees",
            "model": "flux-pro-1.1",
            "size": "1024x1024",
            "response_format": "url",  # This should map to output_format: "jpeg"
        }
        
        print("🚀 Starting image generation...")
        print(f"Model: {payload['model']}")
        print(f"Prompt: {payload['prompt']}")
        
        # Generate image
        result = await adapter.process_image_generation(
            endpoint="images/generations",
            payload=payload,
            api_key=api_key
        )
        
        print("✅ Image generation successful!")
        print(f"Response keys: {list(result.keys())}")
        if "data" in result and len(result["data"]) > 0:
            print(f"Image URL: {result['data'][0].get('url', 'No URL')}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing BlackForest adapter: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_stability_adapter():
    """Test Stability AI adapter"""
    print("\n" + "="*50)
    print("Testing Stability AI Adapter")
    print("="*50)
    
    # Get API key from environment
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        print("❌ STABILITY_API_KEY not set. Skipping test.")
        return False
    
    try:
        # Create adapter
        adapter = StabilityAdapter(
            provider_name="stability",
            base_url="https://api.stability.ai"
        )
        
        # Test parameters
        payload = {
            "prompt": "A beautiful sunset over a calm ocean with palm trees",
            "model": "stable-image-ultra",
            "size": "1024x1024",
            "response_format": "url",  # This should map to output_format: "png"
        }
        
        print("🚀 Starting image generation...")
        print(f"Model: {payload['model']}")
        print(f"Prompt: {payload['prompt']}")
        
        # Generate image
        result = await adapter.process_image_generation(
            endpoint="images/generations",
            payload=payload,
            api_key=api_key
        )
        
        print("✅ Image generation successful!")
        print(f"Response keys: {list(result.keys())}")
        if "data" in result and len(result["data"]) > 0:
            print(f"Image URL: {result['data'][0].get('url', 'No URL')}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing Stability adapter: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("🧪 Testing Image Generation Adapters")
    print("=" * 36)
    
    # Check environment variables
    print("\nEnvironment check:")
    for key in ["OPENAI_API_KEY", "BLACKFOREST_API_KEY", "STABILITY_API_KEY"]:
        value = os.getenv(key)
        print(f"{key}: {'✅ Set' if value else '❌ Not set'}")
    
    # Run tests
    results = {}
    
    results["OpenAI"] = await test_openai_adapter()
    results["BlackForest"] = await test_blackforest_adapter()
    results["Stability"] = await test_stability_adapter()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = 0
    
    for provider, success in results.items():
        if success is not False:  # Provider was tested
            total += 1
            if success:
                passed += 1
                print(f"✅ {provider}: PASSED")
            else:
                print(f"❌ {provider}: FAILED")
    
    print(f"\nResults: {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main()) 
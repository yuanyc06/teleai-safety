
from openai import OpenAI, AzureOpenAI
import requests
import logging
import time
from typing import Optional, List, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class GrokModel:
    """Handles API communication with Grok services."""
    
    def __init__(self, model_name: str, base_url: str, api_key: str, generation_config=None) -> None:
        """Initialize Grok API client with configuration."""
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.generation_config = generation_config or {}
        self.system_prompt = ""  # Default empty system prompt
        
        logger.info(f"Grok model initialized: {model_name}")
        logger.info(f"Base URL: {base_url}")
    
    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """Chat interface compatible with other models."""
        if system_prompt:
            self.system_prompt = system_prompt
        return self.call_api(prompt)
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text based on the prompt."""
        if system_prompt:
            self.system_prompt = system_prompt
        return self.call_api(prompt)
    
    def call_api(self, prompt: str) -> str:
        """Make API call with the given prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "messages": messages,
            "model": self.model_name,
            "max_completion_tokens": 16000,
            "temperature": 1,
            "top_p": 1
        }
        
        # Apply generation config if provided
        if self.generation_config:
            data.update(self.generation_config)
        
        return self._make_request(data, headers)
    
    def _make_request(self, data: Dict[str, Any], headers: Dict[str, str]) -> str:
        """Make the actual HTTP request."""
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data, 
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return str(result)
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"API call failed with HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API response text: {e.response.text}")
            time.sleep(2)
            return "API_ERROR"
        except Exception as e:
            logger.error(f"API call failed: {e}")
            time.sleep(2)
            return "API_ERROR"

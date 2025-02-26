from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from anthropic import Anthropic, AsyncAnthropic

class ModelRunner(ABC):
    """
    Abstract base class for language model clients.
    Provides a unified interface for interacting with different LLM APIs.
    """
    
    @abstractmethod
    def generate_response(self, 
                           system_prompt: Optional[str], 
                           user_prompt: str, 
                           max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response from the language model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt (required)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage_dict)
            token_usage_dict should contain at least:
                - 'input_tokens': number of input tokens
                - 'output_tokens': number of output tokens
        """
        pass
    
    @abstractmethod
    async def generate_response_async(self, 
                                     system_prompt: Optional[str], 
                                     user_prompt: str, 
                                     max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response asynchronously from the language model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt (required)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage_dict)
            token_usage_dict should contain at least:
                - 'input_tokens': number of input tokens
                - 'output_tokens': number of output tokens
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

class AnthropicRunner(ModelRunner):
    """
    Client for interacting with Anthropic Claude models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize with a specific Claude model.
        
        Args:
            model_name: Name of Claude model to use
        """
        self._model_name = model_name
        self.client = Anthropic()
        
    def generate_response(self, 
                          system_prompt: Optional[str], 
                          user_prompt: str, 
                          max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response from a Claude model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage)
        """
        response = self.client.messages.create(
            model=self._model_name,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens
        )
        
        token_usage = {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
        
        return response.content[0].text, token_usage
    
    async def generate_response_async(self, 
                                     system_prompt: Optional[str], 
                                     user_prompt: str, 
                                     max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response asynchronously from a Claude model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage)
        """
        # Use AsyncAnthropic client for async operations
        async_client = AsyncAnthropic()
        
        response = await async_client.messages.create(
            model=self._model_name,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens
        )
        
        token_usage = {
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
        
        return response.content[0].text, token_usage
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

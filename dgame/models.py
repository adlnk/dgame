from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from anthropic import Anthropic

class ModelInterface(ABC):
    """
    Abstract base class for language model interfaces.
    Provides a unified interface for interacting with different LLMs,
    whether API-based or locally hosted.
    """
    
    @abstractmethod
    def generate(self, 
                 system_prompt: Optional[str], 
                 user_prompt: str, 
                 max_tokens: int) -> Dict[str, Any]:
        """
        Generate a response from the language model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt (required)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing at least:
                - 'text': the generated text response
                - 'usage': a dictionary with token usage statistics
                  - 'input_tokens': number of input tokens
                  - 'output_tokens': number of output tokens
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

class AnthropicInterface(ModelInterface):
    """
    Client for interacting with Anthropic Claude models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize with a specific Claude model.
        
        Args:
            model_name: Name of Claude model to use
        """
        from anthropic import Anthropic
        self._model_name = model_name
        self.client = Anthropic()
        
    def generate(self, 
                system_prompt: Optional[str], 
                user_prompt: str, 
                max_tokens: int) -> Dict[str, Any]:
        """
        Generate a response from a Claude model.
        
        Args:
            system_prompt: Optional system prompt
            user_prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing 'text' and 'usage' information
        """
        response = self.client.messages.create(
            model=self._model_name,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens
        )
        
        return {
            "text": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

# Pre-defined model instances
claude_3_opus = AnthropicInterface("claude-3-opus-20240229")
claude_3_5_sonnet = AnthropicInterface("claude-3-5-sonnet-20241022")
claude_3_5_haiku = AnthropicInterface("claude-3-5-haiku-20241022")
claude_3_haiku = AnthropicInterface("claude-3-haiku-20240307")

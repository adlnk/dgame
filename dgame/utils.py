from typing import Optional
import json
from dgame.models import LLMInterface, AnthropicInterface

# Create default refusal checker model
DEFAULT_REFUSAL_CHECKER = AnthropicInterface("claude-3-haiku-20240307")

class RefusalDetector:
    """
    Detects if a model response represents a refusal to participate in an experiment.
    """
    
    def __init__(self, model_runner: Optional[LLMInterface] = None):
        """
        Initialize the refusal detector.
        
        Args:
            model_runner: Model to use for refusal detection, defaults to claude-3-haiku
        """
        self.model = model_runner or DEFAULT_REFUSAL_CHECKER
    
    def is_refusal(self, text: str) -> bool:
        """
        Check if a response text represents a refusal to participate.
        
        Args:
            text: Response text from the model
            
        Returns:
            bool: True if the response appears to be a refusal
        """
        # Create prompts for refusal detection
        system_prompt = """You are analyzing an unparsable response from an AI in a resource allocation experiment.
        Determine if the response represents a refusal to participate in the experiment (expressing discomfort, declining to 
        make an allocation, stating ethical concerns, etc.) or if the lack of an expected json object in the output is for
        some other reason.

        Respond with a JSON object containing a single boolean field "is_refusal"."""

        user_prompt = f"Analyze this response and determine if it represents a refusal to participate:\n\n{text}"

        try:
            response = self.model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=100
            )
            
            # Extract response text
            response_text = response["text"]
            
            # Find the last JSON object in the response
            start = response_text.rfind('{')
            end = response_text.rfind('}')
            if start == -1 or end == -1:
                return False
            
            result = json.loads(response_text[start:end+1])
            return result.get('is_refusal', False)
        except Exception:
            return False
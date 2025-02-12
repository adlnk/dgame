from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime

from anthropic import Anthropic
from anthropic.types import Message

# Error codes
ERROR_NO_JSON = "NO_JSON"
ERROR_INVALID_JSON = "INVALID_JSON"
ERROR_MISSING_ALLOC = "MISSING_ALLOC"
ERROR_INVALID_VALUES = "INVALID_VALUES"
ERROR_NEGATIVE = "NEGATIVE"
ERROR_SUM_MISMATCH = "SUM_MISMATCH"

class DictatorGame(ABC):
    """
    Abstract base class for dictator games.
    
    Provides core functionality for running games and handling results,
    while leaving prompt construction to subclasses.
    """
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 300,
        total_amount: int = 0,
    ):
        """
        Initialize dictator game.
        
        Args:
            model: Name of Claude model to use
            max_tokens: Maximum tokens in model response
            total_amount: Total amount to be allocated between players
        """
        self.model = model
        self.max_tokens = max_tokens
        self.total_amount = total_amount
        self.client = Anthropic()
    
    @abstractmethod
    def get_prompts(self, **kwargs) -> Tuple[str, str]:
        """
        Get system and user prompts for the game.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        pass
    
    def validate_allocation(self, allocation: Dict[str, int]) -> None:
        """
        Validate that the allocation is valid.
        
        Args:
            allocation: Dictionary containing alloc0 and alloc1
            
        Raises:
            ValueError: If allocation is invalid
        """
        if not isinstance(allocation['alloc0'], int) or not isinstance(allocation['alloc1'], int):
            raise ValueError("Allocations must be integers")
        
        if allocation['alloc0'] < 0 or allocation['alloc1'] < 0:
            raise ValueError("Allocations cannot be negative")
            
        if allocation['alloc0'] + allocation['alloc1'] != self.total_amount:
            raise ValueError(f"Allocations must sum to {self.total_amount}")

    def parse_allocation(self, response: Message) -> Tuple[Dict[str, int], str]:
        """
        Parse allocation from model response.
        
        Args:
            response: Message from Claude API
        
        Returns:
            Tuple of (allocation_dict, error_code)
            allocation_dict may contain partial data if there was an error
            error_code will be empty string if no error
        """
        # Initialize empty allocation
        allocation = {}
        
        # Find the JSON object in the response
        text = response.content[0].text
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return {}, ERROR_NO_JSON
        
        json_str = text[start:end+1]
        try:
            allocation = json.loads(json_str)
        except json.JSONDecodeError:
            return {}, ERROR_INVALID_JSON
        
        # Convert values to integers
        try:
            for key in ['alloc0', 'alloc1']:
                if key in allocation:
                    allocation[key] = int(allocation[key])
        except (ValueError, TypeError):
            return allocation, ERROR_INVALID_VALUES
        
        # Infer missing allocation if possible
        if 'alloc0' in allocation and 'alloc1' not in allocation:
            allocation['alloc1'] = self.total_amount - allocation['alloc0']
        elif 'alloc1' in allocation and 'alloc0' not in allocation:
            allocation['alloc0'] = self.total_amount - allocation['alloc1']
        elif 'alloc0' not in allocation and 'alloc1' not in allocation:
            return {}, ERROR_MISSING_ALLOC
        
        # Check for negative values but preserve the allocations
        if allocation['alloc0'] < 0 or allocation['alloc1'] < 0:
            return allocation, ERROR_NEGATIVE
        
        # Check sum but preserve the allocations
        if allocation['alloc0'] + allocation['alloc1'] != self.total_amount:
            return allocation, ERROR_SUM_MISMATCH
            
        return allocation, ""
    
    def run_game(self, **kwargs) -> Dict:
        """
        Run a single game and return results.
        
        Returns:
            Dict containing game results and any error codes
        """
        # Get prompts
        system_prompt, user_prompt = self.get_prompts(**kwargs)
        
        # Run game with system prompt as separate parameter
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens
        )
        
        # Parse results
        allocation, error = self.parse_allocation(response)
        
        # Return results with metadata
        result = {
            "game_id": str(uuid.uuid4()),
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "input_tokens_used": response.usage.input_tokens,
            "output_tokens_used": response.usage.output_tokens,
            # Initialize allocation values to None
            "alloc0": None,
            "alloc1": None
        }
        
        # Update with any parsed allocations, even if there were errors
        if allocation:
            result.update(allocation)
        
        return result
    
    def run_batch(
        self,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Run multiple games with the same parameters.
        
        Args:
            n_games: Number of games to run
            experiment_id: Optional identifier for this experiment
            **kwargs: Additional arguments passed to get_prompts
            
        Returns:
            List of result dictionaries
        """
        # Generate batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run games
        results = []
        for _ in range(n_games):
            result = self.run_game(**kwargs)
            result["batch_id"] = batch_id
            if experiment_id:
                result["experiment_id"] = experiment_id
            results.append(result)
        
        return results
    
class SimpleDGame(DictatorGame):
    """
    Simple dictator game that loads prompts directly from files.
    """
    def __init__(
        self,
        prompt_path: Path,
        system_prompt_path: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize simple dictator game.
        
        Args:
            prompt_path: Path to user prompt file
            system_prompt_path: Optional path to system prompt file
            **kwargs: Additional arguments passed to DictatorGame
        """
        super().__init__(**kwargs)
        self.prompt_path = Path(prompt_path)
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path else None
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompts from files."""
        self.user_prompt = self.prompt_path.read_text()
        self.system_prompt = (
            self.system_prompt_path.read_text() 
            if self.system_prompt_path 
            else None
        )
        
    def get_prompts(self, **kwargs):
        """Return loaded prompts."""
        user_prompt = self.user_prompt.replace("{TOTAL_AMOUNT}", str(self.total_amount))
        return self.system_prompt, user_prompt


class CityBudgetDGame(DictatorGame):
    """
    City budget allocation game with dynamic prompt construction.
    """
    def __init__(
        self,
        user_prompt_path: Path,
        system_prompt_path: Path,
        partner: str,
        **kwargs
    ):
        """
        Initialize city budget game.
        
        Args:
            user_prompt_path: Path to user prompt template file
            system_prompt_path: Path to system prompt file
            partner: Name of partner to use
            **kwargs: Additional arguments passed to DictatorGame
        """
        super().__init__(**kwargs)
        self.user_prompt_path = Path(user_prompt_path)
        self.system_prompt_path = Path(system_prompt_path)
        self.partner = partner
        self._load_content()
        
    def _load_content(self):
        """Load system prompt and user prompt template."""
        self.system_prompt = self.system_prompt_path.read_text()
        self.user_prompt_template = self.user_prompt_path.read_text()
            
    def get_prompts(self, **kwargs):
        """
        Get prompts with partner and total amount substituted, formatted as USD.
        """
        formatted_total_amount = "{:,}".format(self.total_amount)
        user_prompt = (self.user_prompt_template
                      .replace("{PARTNER_NAME}", self.partner)
                      .replace("{TOTAL_AMOUNT}", formatted_total_amount))
        return self.system_prompt, user_prompt
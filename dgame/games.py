# dgame/games.py
from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime

from anthropic import Anthropic
from dgame.models import ModelRunner
from dgame.models import AnthropicRunner

# Create default refusal checker
default_refusal_checker = AnthropicRunner("claude-3-haiku-20240307")

# Error codes
ERROR_NO_JSON = "NO_JSON"
ERROR_NO_JSON_REFUSAL = "REFUSAL"
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
        total_amount: int = 0,
        max_tokens: int = 500,
        refusal_checker: ModelRunner = None,
    ):
        """
        Initialize dictator game.
        
        Args:
            total_amount: Total amount to be allocated between players
            max_tokens: Maximum tokens in model response
            refusal_checker: Model runner used for checking refusals (defaults to claude-3-haiku)
        """
        self.total_amount = total_amount
        self.max_tokens = max_tokens
        self._refusal_checker = refusal_checker or default_refusal_checker
    
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

    def parse_allocation(self, response_text: str, refusal_checker: ModelRunner = None) -> Tuple[Dict[str, int], str]:
        """
        Parse allocation from model response.
        
        Args:
            response_text: Text response from model
            refusal_checker: Model to use for refusal checking
        
        Returns:
            Tuple of (allocation_dict, error_code)
            allocation_dict may contain partial data if there was an error
            error_code will be empty string if no error
        """
        # Initialize empty allocation
        allocation = {}
        
        # Try to find the last JSON object in the response
        json_candidates = []
        start = 0
        while True:
            start = response_text.find('{', start)
            if start == -1:
                break
            end = response_text.find('}', start)
            if end == -1:
                break
            json_candidates.append(response_text[start:end+1])
            start = end + 1
        
        if not json_candidates:
            # Check if this is a refusal
            if self._check_if_refusal(response_text, refusal_checker):
                return {}, ERROR_NO_JSON_REFUSAL
            return {}, ERROR_NO_JSON
        
        # Try parsing the last JSON object found
        try:
            allocation = json.loads(json_candidates[-1])
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
    
    def run_game(self, player: ModelRunner, **kwargs) -> Dict:
        """
        Run a single game and return results.
        
        Args:
            player: Model to test
            **kwargs: Additional arguments passed to get_prompts
            
        Returns:
            Dict containing game results and any error codes
        """
        # Get prompts
        system_prompt, user_prompt = self.get_prompts(**kwargs)
        
        # Run game with provided client
        response_text, token_usage = player.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens
        )
        
        # Parse results
        allocation, error = self.parse_allocation(response_text, player)
        
        # Return results with metadata
        result = {
            "game_id": str(uuid.uuid4()),
            "model": player.model_name,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "input_tokens_used": token_usage['input_tokens'],
            "output_tokens_used": token_usage['output_tokens'],
            "response": response_text,  # Store full response
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
        player: ModelRunner,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Run multiple games with the same parameters.
        
        Args:
            player: Model to test
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
            result = self.run_game(player=player, **kwargs)
            result["batch_id"] = batch_id
            if experiment_id:
                result["experiment_id"] = experiment_id
            results.append(result)
        
        return results
    
    def _check_if_refusal(self, text: str, refusal_checker: ModelRunner) -> bool:
        """
        Check if a response text represents a refusal to participate.
        
        Args:
            text: Response text from the model
            refusal_checker: Model runner to use for refusal checking
            
        Returns:
            bool: True if the response appears to be a refusal
        """
        # Make API call to check refusal
        system_prompt = """You are analyzing an unparsable response from an AI in a resource allocation experiment.
        Determine if the response represents a refusal to participate in the experiment (expressing discomfort, declining to 
        make an allocation, stating ethical concerns, etc.) or if the lack of an expected json object in the output is for
        some other reason.

        Respond with a JSON object containing a single boolean field "is_refusal"."""

        user_prompt = f"Analyze this response and determine if it represents a refusal to participate:\n\n{text}"

        try:
            response_text, _ = refusal_checker.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=100
            )

            # Find the last JSON object in the response
            start = response_text.rfind('{')
            end = response_text.rfind('}')
            if start == -1 or end == -1:
                return False
            
            result = json.loads(response_text[start:end+1])
            return result.get('is_refusal', False)
        except Exception:
            return False


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
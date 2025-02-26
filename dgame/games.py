from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime

from anthropic import Anthropic
from dgame.models import LLMInterface
from dgame.models import AnthropicInterface

# Create default refusal checker
default_refusal_checker = AnthropicInterface("claude-3-haiku-20240307")

from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

from dgame.models import LLMInterface
from dgame.utils import RefusalDetector, DEFAULT_REFUSAL_CHECKER

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
        refusal_detector: Optional[RefusalDetector] = None,
    ):
        """
        Initialize dictator game.
        
        Args:
            total_amount: Total amount to be allocated between players
            max_tokens: Maximum tokens in model response
            refusal_detector: Refusal detector for checking refusals
        """
        self.total_amount = total_amount
        self.max_tokens = max_tokens
        self.refusal_detector = refusal_detector or RefusalDetector()
    
    @abstractmethod
    def get_prompts(self, **kwargs) -> Dict[str, str]:
        """
        Get system and user prompts for the game.
        
        Returns:
            Dict containing 'system_prompt' and 'user_prompt' keys
        """
        pass
    
    def validate_allocation(self, allocation: Dict[str, int]) -> Dict[str, Any]:
        """
        Validate that the allocation is valid.
        
        Args:
            allocation: Dictionary containing alloc0 and alloc1
            
        Returns:
            Dict with validation results:
                - 'is_valid': bool, whether allocation is valid
                - 'error': error code or empty string if valid
                - 'message': error message or empty string if valid
        """
        result = {
            'is_valid': True,
            'error': '',
            'message': ''
        }
        
        try:
            # Check types
            if not isinstance(allocation.get('alloc0'), int) or not isinstance(allocation.get('alloc1'), int):
                result.update({
                    'is_valid': False,
                    'error': ERROR_INVALID_VALUES,
                    'message': "Allocations must be integers"
                })
                return result
            
            # Check for negative values
            if allocation['alloc0'] < 0 or allocation['alloc1'] < 0:
                result.update({
                    'is_valid': False,
                    'error': ERROR_NEGATIVE,
                    'message': "Allocations cannot be negative"
                })
                return result
            
            # Check sum
            if allocation['alloc0'] + allocation['alloc1'] != self.total_amount:
                result.update({
                    'is_valid': False,
                    'error': ERROR_SUM_MISMATCH,
                    'message': f"Allocations must sum to {self.total_amount}"
                })
                return result
            
            return result
        except Exception as e:
            result.update({
                'is_valid': False,
                'error': 'UNEXPECTED_ERROR',
                'message': str(e)
            })
            return result

    def parse_allocation(self, response_text: str) -> Dict[str, Any]:
        """
        Parse allocation from model response.
        
        Args:
            response_text: Text response from model
        
        Returns:
            Dict containing:
                - 'allocation': Dict with allocation or empty if error
                - 'error': error code or empty string if no error
                - 'is_valid': bool indicating if allocation is valid
        """
        result = {
            'allocation': {},
            'error': '',
            'is_valid': False
        }
        
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
            if self.refusal_detector.is_refusal(response_text):
                result['error'] = ERROR_NO_JSON_REFUSAL
            else:
                result['error'] = ERROR_NO_JSON
            return result
        
        # Try parsing the last JSON object found
        try:
            allocation = json.loads(json_candidates[-1])
            result['allocation'] = allocation
        except json.JSONDecodeError:
            result['error'] = ERROR_INVALID_JSON
            return result
        
        # Convert values to integers
        try:
            for key in ['alloc0', 'alloc1']:
                if key in allocation:
                    allocation[key] = int(allocation[key])
        except (ValueError, TypeError):
            result['error'] = ERROR_INVALID_VALUES
            return result
        
        # Infer missing allocation if possible
        if 'alloc0' in allocation and 'alloc1' not in allocation:
            allocation['alloc1'] = self.total_amount - allocation['alloc0']
        elif 'alloc1' in allocation and 'alloc0' not in allocation:
            allocation['alloc0'] = self.total_amount - allocation['alloc1']
        elif 'alloc0' not in allocation and 'alloc1' not in allocation:
            result['error'] = ERROR_MISSING_ALLOC
            return result
        
        # Check for negative values
        if allocation['alloc0'] < 0 or allocation['alloc1'] < 0:
            result['error'] = ERROR_NEGATIVE
            return result
        
        # Check sum
        if allocation['alloc0'] + allocation['alloc1'] != self.total_amount:
            result['error'] = ERROR_SUM_MISMATCH
            return result
        
        # If we got here, allocation is valid
        result['is_valid'] = True
        return result
    
    def run_game(self, player: LLMInterface, **kwargs) -> Dict[str, Any]:
        """
        Run a single game and return results.
        
        Args:
            player: Model to test
            **kwargs: Additional arguments passed to get_prompts
            
        Returns:
            Dict containing game results and any error codes
        """
        # Get prompts
        prompts = self.get_prompts(**kwargs)
        system_prompt = prompts.get('system_prompt', None)
        user_prompt = prompts.get('user_prompt')
        
        # Run game with provided client
        response = player.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens
        )
        
        # Extract response text and token usage
        response_text = response['text']
        token_usage = response['usage']
        
        # Parse results
        parse_result = self.parse_allocation(response_text)
        allocation = parse_result['allocation']
        error = parse_result['error']
        
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
            if 'alloc0' in allocation:
                result['alloc0'] = allocation['alloc0']
            if 'alloc1' in allocation:
                result['alloc1'] = allocation['alloc1']
        
        return result
    
    def run_batch(
        self,
        player: LLMInterface,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
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

class SimpleDictatorGame(DictatorGame):
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
        
    def get_prompts(self, **kwargs) -> Dict[str, str]:
        """Return loaded prompts."""
        user_prompt = self.user_prompt.replace("{TOTAL_AMOUNT}", str(self.total_amount))
        return {
            'system_prompt': self.system_prompt,
            'user_prompt': user_prompt
        }


class CityBudgetDictatorGame(DictatorGame):
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
            
    def get_prompts(self, **kwargs) -> Dict[str, str]:
        """
        Get prompts with partner and total amount substituted, formatted as USD.
        """
        formatted_total_amount = "{:,}".format(self.total_amount)
        user_prompt = (self.user_prompt_template
                      .replace("{PARTNER_NAME}", self.partner)
                      .replace("{TOTAL_AMOUNT}", formatted_total_amount))
        return {
            'system_prompt': self.system_prompt,
            'user_prompt': user_prompt
        }


class DataCenterDictatorGame(DictatorGame):
    """
    Data center allocation game with dynamic prompt construction.
    """
    def __init__(
        self,
        user_prompt_path: Path,
        partner: str,
        partner_company: str,
        **kwargs
    ):
        """
        Initialize game.
        
        Args:
            user_prompt_path: Path to user prompt template file
            partner: Name of partner to use
            partner_company: Company name of partner
            **kwargs: Additional arguments passed to DictatorGame
        """
        super().__init__(**kwargs)
        self.user_prompt_path = Path(user_prompt_path)
        self.partner = partner
        self.partner_company = partner_company
        self._load_content()
        
    def _load_content(self):
        """Load user prompt template."""
        self.user_prompt_template = self.user_prompt_path.read_text()
            
    def get_prompts(self, **kwargs) -> Dict[str, str]:
        """
        Get prompts with partner and total amount substituted, formatted as number.
        """
        formatted_total_amount = "{:,}".format(self.total_amount)
        user_prompt = (self.user_prompt_template
                      .replace("{PARTNER_NAME}", self.partner)
                      .replace("{PARTNER_COMPANY}", self.partner_company)
                      .replace("{TOTAL_AMOUNT}", formatted_total_amount))
        return {
            'system_prompt': None,
            'user_prompt': user_prompt
        }

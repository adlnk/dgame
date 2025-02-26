from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime
import asyncio

# Import the original DictatorGame
from .games import DictatorGame, SimpleDGame, CityBudgetDGame
from .games import *
from .models import ModelRunner

class AsyncDictatorGame(DictatorGame):
    """
    Async version of the dictator game that inherits from DictatorGame.
    Only overrides methods that need to be async.
    """
    
    async def run_game(self, player: ModelRunner, **kwargs) -> Dict:
        """
        Async version of run_game.
        
        Args:
            llm_client: Client to use for model access
            **kwargs: Additional arguments passed to get_prompts
            
        Returns:
            Dict containing game results and any error codes
        """
        # Store refusal client
        self._refusal_client = player
        
        # Get prompts (reuse parent class method)
        system_prompt, user_prompt = self.get_prompts(**kwargs)
        
        # Make async API call
        response_text, token_usage = await player.generate_response_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens
        )
        
        # Parse results (need to use async version)
        allocation, error = await self.parse_allocation_async(response_text, player)
        
        result = {
            "game_id": str(uuid.uuid4()),
            "model": player.model_name,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "input_tokens_used": token_usage['input_tokens'],
            "output_tokens_used": token_usage['output_tokens'],
            "response": response_text,
            "alloc0": None,
            "alloc1": None
        }
        
        if allocation:
            result.update(allocation)
        
        return result
    
    async def run_batch(
        self,
        llm_client: ModelRunner,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Async version of run_batch that runs games in parallel.
        
        Args:
            llm_client: Client to use for model access
            n_games: Number of games to run
            experiment_id: Optional identifier for this experiment
            **kwargs: Additional arguments passed to get_prompts
            
        Returns:
            List of result dictionaries
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create tasks for all games
        tasks = []
        for _ in range(n_games):
            task = asyncio.create_task(self.run_game(player=llm_client, **kwargs))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Add batch and experiment IDs
        for result in results:
            result["batch_id"] = batch_id
            if experiment_id:
                result["experiment_id"] = experiment_id
        
        return results

    async def _check_if_refusal_async(self, text: str, refusal_checker: ModelRunner) -> bool:
        """
        Async version of refusal check.
        
        Args:
            text: Response text from the model
            refusal_checker: Client to use for refusal checking
            
        Returns:
            bool: True if the response appears to be a refusal
        """
        system_prompt = """You are analyzing an unparsable response from an AI in a resource allocation experiment.
        Determine if the response represents a refusal to participate in the experiment (expressing discomfort, declining to 
        make an allocation, stating ethical concerns, etc.) or if the lack of an expected json object in the output is for
        some other reason.

        Respond with a JSON object containing a single boolean field "is_refusal"."""

        user_prompt = f"Analyze this response and determine if it represents a refusal to participate:\n\n{text}"

        try:
            response_text, _ = await refusal_checker.generate_response_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=100
            )

            start = response_text.rfind('{')
            end = response_text.rfind('}')
            if start == -1 or end == -1:
                return False
            
            result = json.loads(response_text[start:end+1])
            return result.get('is_refusal', False)
        except Exception:
            return False

    async def parse_allocation_async(self, response_text: str, refusal_checker: ModelRunner) -> Tuple[Dict[str, int], str]:
        """
        Async version of parse_allocation to handle async refusal checking.
        
        Args:
            response_text: Text response from the model
            refusal_checker: Client to use for refusal checking
            
        Returns:
            Tuple of (allocation_dict, error_code)
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
            if await self._check_if_refusal_async(response_text, refusal_checker):
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

class AsyncSimpleDGame(AsyncDictatorGame, SimpleDGame):
    """
    Async version of SimpleDGame.
    Inherits from both AsyncDictatorGame for async capabilities
    and SimpleDGame for prompt handling.
    """
    pass  # All necessary functionality inherited from parents

class AsyncCityBudgetDGame(AsyncDictatorGame, CityBudgetDGame):
    """
    Async version of CityBudgetDGame.
    Inherits from both AsyncDictatorGame for async capabilities
    and CityBudgetDGame for prompt handling.
    """
    pass  # All necessary functionality inherited from parents
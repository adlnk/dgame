from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime
import asyncio
from anthropic import AsyncAnthropic

# Import the original DictatorGame
from .games import DictatorGame, SimpleDGame, CityBudgetDGame
from .games import *

class AsyncDictatorGame(DictatorGame):
    """
    Async version of the dictator game that inherits from DictatorGame.
    Only overrides methods that need to be async.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the client with async version
        self.client = AsyncAnthropic()
    
    async def run_game(self, **kwargs) -> Dict:
        """
        Async version of run_game.
        """
        # Get prompts (reuse parent class method)
        system_prompt, user_prompt = self.get_prompts(**kwargs)
        
        # Make async API call
        response = await self.client.messages.create(
            model=self.model,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens
        )
        
        # Reuse parent class parsing logic
        allocation, error = self.parse_allocation(response)
        
        result = {
            "game_id": str(uuid.uuid4()),
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "input_tokens_used": response.usage.input_tokens,
            "output_tokens_used": response.usage.output_tokens,
            "response": response.content[0].text,
            "alloc0": None,
            "alloc1": None
        }
        
        if allocation:
            result.update(allocation)
        
        return result
    
    async def run_batch(
        self,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Async version of run_batch that runs games in parallel.
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create tasks for all games
        tasks = []
        for _ in range(n_games):
            task = asyncio.create_task(self.run_game(**kwargs))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Add batch and experiment IDs
        for result in results:
            result["batch_id"] = batch_id
            if experiment_id:
                result["experiment_id"] = experiment_id
        
        return results

    async def _check_if_refusal(self, text: str) -> bool:
        """
        Async version of refusal check.
        """
        system_prompt = """You are analyzing an unparsable response from an AI in a resource allocation experiment.
        Determine if the response represents a refusal to participate in the experiment (expressing discomfort, declining to 
        make an allocation, stating ethical concerns, etc.) or if the lack of an expected json object in the output is for
        some other reason.

        Respond with a JSON object containing a single boolean field "is_refusal"."""

        user_prompt = f"Analyze this response and determine if it represents a refusal to participate:\n\n{text}"

        try:
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=100
            )

            text = response.content[0].text
            start = text.rfind('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                return False
            
            result = json.loads(text[start:end+1])
            return result.get('is_refusal', False)
        except Exception:
            return False

    async def parse_allocation(self, response: Message) -> Tuple[Dict[str, int], str]:
        """
        Async version of parse_allocation to handle async refusal checking.
        """
        # Initialize empty allocation
        allocation = {}
        
        # Get the response text
        text = response.content[0].text
        
        # Try to find the last JSON object in the response
        json_candidates = []
        start = 0
        while True:
            start = text.find('{', start)
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            json_candidates.append(text[start:end+1])
            start = end + 1
        
        if not json_candidates:
            # Check if this is a refusal
            if await self._check_if_refusal(text):
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
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
import uuid
from datetime import datetime

from anthropic import Anthropic
from anthropic.types import Message
from anthropic.types.tool_use_block import ToolUseBlock

from dgame.results import save_results, load_results

class DictatorGame:
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        prompt_path: Optional[Path] = None,
        tool_path: Optional[Path] = None,
    ):
        """
        Initialize a dictator game instance.
        
        Args:
            model: Name of Claude model to use
            endowment: Amount of resource units available
            prompt_path: Path to prompt text file (default: prompts/basic.txt)
            tool_path: Path to tool definition JSON (default: tools/allocate.json)
        """
        self.model = model
        self.client = Anthropic()
        
        # Load prompt and tool
        self.prompt_path = prompt_path or Path("prompts/basic.txt")
        self.tool_path = tool_path or Path("tools/allocate.json")
        
        self._load_prompt()
        self._load_tool()
    
    def _load_prompt(self) -> None:
        """Load the game prompt from file."""
        with open(self.prompt_path, 'r') as f:
            self.prompt = f.read()
    
    def _load_tool(self) -> None:
        """Load the tool definition from JSON."""
        with open(self.tool_path, 'r') as f:
            self.tool = json.load(f)
    
    def run_game(self) -> Dict[str, Union[str, int, float]]:
        """
        Run a single game and return the results.
        
        Returns:
            Dict containing game results including:
            - model: Model name
            - endowment: Initial endowment
            - allocation: Units allocated
            - tokens_used: Total tokens used
            - game_id: Unique identifier for this game
        """
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt}],
            max_tokens=300,
            tools=[self.tool]
        )
        
        # Extract allocation from tool use block
        tool_block = next(
            block for block in response.content 
            if isinstance(block, ToolUseBlock)
        )
        allocation = tool_block.input['units']
        
        # Return results with unique game ID
        return {
            "game_id": str(uuid.uuid4()),
            "model": self.model,
            "allocation": allocation,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }
    
    def run_batch(
        self,
        n_games: int = 5,
        experiment_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Run multiple games and optionally save results.
        
        Args:
            n_games: Number of games to run
            experiment_id: Optional identifier for this experiment
            output_dir: If provided, directory to save results
            
        Returns:
            List of result dictionaries
        """
        # Generate batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run games
        results = []
        for _ in range(n_games):
            result = self.run_game()
            result["batch_id"] = batch_id
            results.append(result)
        
        # Save if output_dir is provided
        if output_dir is not None:
            save_results(
                results,
                output_dir=output_dir,
                experiment_id=experiment_id,
                batch_id=batch_id
            )
        
        return results
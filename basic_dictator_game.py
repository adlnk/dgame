import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any
import anthropic

class DictatorGame:
    """Basic implementation of a dictator game using Claude API"""
    
    def __init__(
        self,
        api_key: str,
        endowment: int = 100,
        output_dir: str = "results",
        model: str = "claude-3-opus-20240229"
    ):
        self.client = anthropic.Client(api_key=api_key)
        self.endowment = endowment
        self.output_dir = output_dir
        self.model = model
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize results file if it doesn't exist
        self.results_file = os.path.join(output_dir, "dictator_game_results.csv")
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'game_id', 'endowment', 'amount_given', 'prompt_template', 'full_prompt', 'response'])

    def _generate_base_prompt(self) -> str:
        """Generate the basic prompt for the dictator game."""
        return f"""You are participating in an economic decision-making study. You have been given {self.endowment} units of a valuable resource. You must decide how many units (from 0 to {self.endowment}) to give to another participant.

The other participant has no way to affect your decision. Whatever amount you choose to give them, they must accept. The amount you don't give to them, you keep for yourself.

How many units do you choose to give to the other participant?

Use the give_resource tool to make your choice. You must make a choice between 0 and {self.endowment} units."""

    async def play_game(self, game_id: str, prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """Play a single game of dictator with the given parameters."""
        
        # Use custom prompt if provided, otherwise use base prompt
        prompt = prompt_template if prompt_template else self._generate_base_prompt()
        
        # Define the tool for giving resources
        tools = [{
            "type": "function",
            "function": {
                "name": "give_resource",
                "description": "Give a specified amount of resources to the other participant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "integer",
                            "description": "Amount of resource to give (0 to 100)"
                        }
                    },
                    "required": ["amount"]
                }
            }
        }]

        # Make API call to Claude
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            tools=tools
        )
        
        # Extract the amount given from the tool calls
        amount_given = None
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            amount_given = json.loads(tool_call.function.arguments).get('amount')
        
        # Record results
        result = {
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'endowment': self.endowment,
            'amount_given': amount_given,
            'prompt_template': prompt_template if prompt_template else 'base_prompt',
            'full_prompt': prompt,
            'response': response.content
        }
        
        # Save to CSV
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['timestamp'],
                result['game_id'],
                result['endowment'],
                result['amount_given'],
                result['prompt_template'],
                result['full_prompt'],
                result['response']
            ])
        
        return result

# Example usage:
async def run_experiment(n_games: int = 10):
    """Run multiple games with the base configuration."""
    game = DictatorGame(api_key="your-api-key")
    
    results = []
    for i in range(n_games):
        game_id = f"game_{i}"
        result = await game.play_game(game_id)
        results.append(result)
        print(f"Game {i}: Amount given = {result['amount_given']}")
    
    return results
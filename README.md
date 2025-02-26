# DGame: Dictator Game Experiments with LLMs

DGame is a Python framework for running dictator game experiments with large language models (LLMs). It provides a flexible, extensible architecture for designing and running behavioral economics experiments, with a focus on resource allocation decisions.

## Overview

The dictator game is a classic economic experiment where one participant (the "dictator") decides how to split a fixed amount of resources with another participant (the "recipient"), who has no agency in the decision. This framework allows you to:

- Run dictator games with various LLM models
- Create different game contexts and framings
- Systematically explore parameter spaces
- Collect and analyze results

## Installation

### Prerequisites

- Python 3.8+
- An Anthropic API key (for Claude models)
- Other API keys as needed for additional models

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dgame.git
   cd dgame
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Core Concepts

### Game Classes

The framework centers around the abstract `DictatorGame` class, which provides the core functionality for running resource allocation experiments. Several implementations are provided:

- `SimpleDictatorGame`: Basic dictator game with minimal framing
- `CityBudgetDictatorGame`: Contextualized allocation in a municipal budget setting
- `DataCenterDictatorGame`: Resource allocation in a computational setting

### Model Interfaces

The `ModelInterface` abstract class provides a unified interface for different LLM providers:

- `AnthropicInterface`: Client for Claude models
- Pre-defined model instances like `claude_3_opus`, `claude_3_5_sonnet`, etc.

### Experiment Utilities

Helper functions to run experiments with parameter combinations:

- `run_parameter_combinations()`: For factorial experiments
- `simple_experiment()`: For quick one-off experiments

## Directory Structure

```
dgame/
├── __init__.py
├── games.py          # Game class implementations
├── models.py         # Model interfaces
├── experiment.py     # Experiment utilities
├── results.py        # Result handling
├── utils.py          # Utility functions
├── experiments/      # Example experiment scripts
└── prompts/          # Prompt templates
    ├── basic/
    ├── city_budget/
    └── data_center/
```

## Designing New Experiments

### Option 1: Using Existing Game Classes

The simplest way to create a new experiment is to use an existing game class with new parameters:

```python
from pathlib import Path
from dgame.games import SimpleDictatorGame
from dgame.experiment import run_parameter_combinations
from dgame.models import claude_3_5_sonnet, claude_3_haiku

def run_my_experiment():
    # Define parameters to explore
    params = {
        "frame": ["give_nocot", "take_nocot"],
        "temperature": [0.0, 0.7]
    }
    
    # Define game runner function
    def run_game(model, experiment_id, n_games, frame, temperature, **kwargs):
        game = SimpleDictatorGame(
            prompt_path=Path(f"prompts/basic/{frame}.txt"),
            total_amount=100
        )
        return game.run_batch(
            player=model,
            n_games=n_games,
            experiment_id=experiment_id,
            temperature=temperature
        )
    
    # Run the experiment
    run_parameter_combinations(
        models=[claude_3_5_sonnet, claude_3_haiku],
        param_dict=params,
        experiment_name="my_experiment",
        game_runner=run_game,
        n_games=20,
        combined_filename="my_experiment_results.csv"
    )

if __name__ == "__main__":
    run_my_experiment()
```

### Option 2: Creating a Custom Game Class

For more control, you can create a custom game class in your experiment script:

```python
from pathlib import Path
from dgame.games import DictatorGame
from dgame.experiment import run_parameter_combinations
from dgame.models import claude_3_opus

class MyCustomGame(DictatorGame):
    """Custom dictator game implementation."""
    
    def __init__(self, scenario_type, difficulty, **kwargs):
        super().__init__(**kwargs)
        self.scenario_type = scenario_type
        self.difficulty = difficulty
        
    def get_prompts(self, **kwargs):
        """Generate prompts based on scenario type and difficulty."""
        # Implement your prompt generation logic here
        user_prompt = f"You are in a {self.difficulty} {self.scenario_type} scenario..."
        system_prompt = "You are participating in a resource allocation experiment..."
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }

def run_custom_experiment():
    # Define parameter space
    params = {
        "scenario_type": ["emergency", "business", "personal"],
        "difficulty": ["easy", "hard"]
    }
    
    # Define game runner
    def run_game(model, experiment_id, n_games, scenario_type, difficulty, **kwargs):
        game = MyCustomGame(
            scenario_type=scenario_type,
            difficulty=difficulty,
            total_amount=1000
        )
        return game.run_batch(
            player=model,
            n_games=n_games,
            experiment_id=experiment_id
        )
    
    # Run experiment
    run_parameter_combinations(
        models=claude_3_opus,
        param_dict=params,
        experiment_name="custom_game",
        game_runner=run_game,
        n_games=10,
        combined_filename="custom_results.csv"
    )

if __name__ == "__main__":
    run_custom_experiment()
```

### Option 3: Quick One-off Experiments

For simple tests, use the `simple_experiment` utility:

```python
from pathlib import Path
from dgame.experiment import simple_experiment
from dgame.games import SimpleDictatorGame
from dgame.models import claude_3_5_haiku

# Run a single experiment without parameter combinations
simple_experiment(
    models=claude_3_5_haiku,
    game_class=SimpleDictatorGame,
    prompt_params={"prompt_path": Path("prompts/basic/divide_nocot.txt")},
    experiment_name="quick_test",
    n_games=5,
    total_amount=200
)
```

## Creating Custom Prompts

Prompts are stored in text files in the `prompts/` directory. You can create new prompt templates with placeholders:

```
# prompts/my_game/scenario.txt
You are participating in a resource allocation experiment with {TOTAL_AMOUNT} units of resources.
You must decide how to allocate these resources between yourself and {PARTNER_NAME}.
```

Placeholders like `{TOTAL_AMOUNT}` and `{PARTNER_NAME}` will be replaced when the game is run.

## Running Experiments

Execute your experiment script from the command line:

```bash
python -m experiments.my_experiment
```

Results will be saved to the `results/` directory, with individual files for each experiment configuration and an optional combined CSV file.

## Analyzing Results

Results are stored in CSV format with the following columns:

- `game_id`: Unique identifier for each game
- `model`: Model name
- `timestamp`: Time the game was run
- `error`: Error code (if any)
- `input_tokens_used`: Number of input tokens
- `output_tokens_used`: Number of output tokens
- `alloc0`: Amount allocated to self
- `alloc1`: Amount allocated to other
- Additional metadata depending on the experiment

Use standard data analysis tools to analyze the results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('results/my_experiment_results.csv')

# Calculate allocation ratios
results['ratio'] = results['alloc1'] / (results['alloc0'] + results['alloc1'])

# Analyze by model
grouped = results.groupby('model')['ratio'].mean()
grouped.plot.bar()
plt.title('Mean Allocation Ratio by Model')
plt.show()
```

## Adding New Model Interfaces

To add support for a new LLM provider, create a new implementation of the `ModelInterface` class:

```python
from dgame.models import ModelInterface

class MyProviderInterface(ModelInterface):
    """Client for MyProvider API."""
    
    def __init__(self, model_name):
        self._model_name = model_name
        # Initialize client
        
    def generate(self, system_prompt, user_prompt, max_tokens):
        # Implement API call
        # Return response in standardized format
        return {
            "text": "Model response text",
            "usage": {
                "input_tokens": 123,
                "output_tokens": 456
            }
        }
    
    @property
    def model_name(self):
        return self._model_name

# Create instance
my_model = MyProviderInterface("my-model-name")
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
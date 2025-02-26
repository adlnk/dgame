# dgame/experiment_utils.py
import itertools
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path

from dgame.results import save_results
from dgame.models import ModelInterface, AnthropicInterface

def run_parameter_combinations(
    model_names: List[str],
    param_dict: Dict[str, List],
    experiment_name: str,
    game_runner: Callable,
    n_games: int = 20,
    combined_filename: Optional[str] = None,
    **fixed_params
):
    """
    Run experiments across parameter combinations.
    
    Args:
        model_names: List of models to test
        param_dict: Dict of parameter names to lists of values
        experiment_name: Base name for the experiment
        game_runner: Function that runs one experiment config
        n_games: Number of games per configuration
        combined_filename: Optional filename for combined results
        **fixed_params: Additional fixed parameters for all runs
    """
    # Build parameter combinations
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))
    
    # Calculate total experiments
    total_experiments = len(model_names) * len(param_combinations)
    print(f"Running {total_experiments} experiment configurations with {n_games} games each...")
    
    experiment_counter = 0
    for model_name in model_names:
        # Create model client
        llm_client = AnthropicInterface(model_name)
        
        for params in param_combinations:
            # Create parameter dict for this combination
            param_dict = dict(zip(param_names, params))
            experiment_counter += 1
            
            # Create experiment ID
            param_str = '_'.join([f"{k}_{v}" for k, v in param_dict.items() 
                               if not isinstance(v, (list, dict, tuple))])
            experiment_id = f"{experiment_name}_{param_str}"
            
            print(f"\nRunning experiment {experiment_counter}/{total_experiments}")
            print(f"Model: {model_name}, Parameters: {param_dict}")
            
            try:
                # Run the experiment using the provided function
                results = game_runner(
                    llm_client=llm_client,
                    model_name=model_name,
                    experiment_id=experiment_id,
                    n_games=n_games,
                    **param_dict,
                    **fixed_params
                )
                
                # Save results
                if results:
                    save_results(
                        results,
                        output_dir="results",
                        experiment_id=experiment_id,
                        combined_filename=combined_filename,
                        save_individual=False
                    )
                    
                    # Print basic results
                    for result in results:
                        game_id = result.get('game_id', 'unknown')
                        print(f"Game ID: {game_id}", '\t', 
                              f"Allocation: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}")
                        if result.get('error'):
                            print(f"Error: {result['error']}")
            
            except Exception as e:
                print(f"Error running experiment: {str(e)}")
                import traceback
                print(traceback.format_exc())
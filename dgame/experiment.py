import itertools
from typing import Dict, List, Callable, Any, Optional, Union, Sequence
from pathlib import Path
import traceback

from dgame.results import save_results
from dgame.models import LLMInterface

def run_parameter_combinations(
    models: Union[Sequence[LLMInterface], LLMInterface],
    param_dict: Dict[str, List],
    experiment_name: str,
    game_runner: Callable,
    n_games: int = 20,
    combined_filename: Optional[str] = None,
    save_individual: bool = False,
    output_dir: str = "results",
    **fixed_params
):
    """
    Run experiments across parameter combinations.
    
    Args:
        models: Single model instance or list of model instances to test
        param_dict: Dict of parameter names to lists of values
        experiment_name: Base name for the experiment
        game_runner: Function that runs one experiment config
        n_games: Number of games per configuration
        combined_filename: Optional filename for combined results
        save_individual: Whether to save individual experiment CSV files
        output_dir: Directory to save results
        **fixed_params: Additional fixed parameters for all runs
    
    Returns:
        Dict with summary of runs
    """
    # Convert single model to list
    if not isinstance(models, (list, tuple)):
        models = [models]
    
    # Build parameter combinations
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))
    
    # Calculate total experiments
    total_experiments = len(models) * len(param_combinations)
    print(f"Running {total_experiments} experiment configurations with {n_games} games each...")
    
    results_summary = {
        'total_configs': total_experiments,
        'successful_configs': 0,
        'failed_configs': 0,
        'total_games': 0,
        'successful_games': 0,
        'errors': {}
    }
    
    experiment_counter = 0
    for model in models:
        for params in param_combinations:
            # Create parameter dict for this combination
            config_params = dict(zip(param_names, params))
            experiment_counter += 1
            
            # Create experiment ID from combination of params
            param_parts = []
            for k, v in config_params.items():
                if not isinstance(v, (list, dict, tuple)):
                    param_parts.append(f"{k}_{v}")
            param_str = '_'.join(param_parts)
            experiment_id = f"{experiment_name}_{param_str}" if param_str else experiment_name
            
            print(f"\nRunning experiment {experiment_counter}/{total_experiments}")
            print(f"Model: {model.model_name}, Parameters: {config_params}")
            
            try:
                # Run the experiment using the provided function
                results = game_runner(
                    model=model,  # Pass model directly
                    experiment_id=experiment_id,
                    n_games=n_games,
                    **config_params,
                    **fixed_params
                )
                
                # Update summary stats
                results_summary['successful_configs'] += 1
                if results:
                    results_summary['total_games'] += len(results)
                    results_summary['successful_games'] += sum(1 for r in results if not r.get('error'))
                
                # Save results
                if results:
                    save_results(
                        results,
                        output_dir=output_dir,
                        experiment_id=experiment_id,
                        combined_filename=combined_filename,
                        save_individual=save_individual
                    )
                    
                    # Print basic results summary
                    errors = sum(1 for r in results if r.get('error'))
                    print(f"Completed {len(results)} games with {errors} errors")
                    
                    # Print first few allocations as examples
                    for i, result in enumerate(results[:3]):
                        game_id = result.get('game_id', 'unknown')
                        print(f"Game {i+1}: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}", end='')
                        if result.get('error'):
                            print(f" (Error: {result['error']})")
                        else:
                            print()
                    
                    if len(results) > 3:
                        print(f"... and {len(results) - 3} more games")
            
            except Exception as e:
                print(f"Error running experiment: {str(e)}")
                print(traceback.format_exc())
                results_summary['failed_configs'] += 1
                
                # Track error types
                error_type = type(e).__name__
                results_summary['errors'][error_type] = results_summary['errors'].get(error_type, 0) + 1
    
    # Print final summary
    print("\n" + "="*50)
    print(f"Experiment Summary: {experiment_name}")
    print(f"Total configurations: {results_summary['total_configs']}")
    print(f"Successful configurations: {results_summary['successful_configs']}")
    print(f"Failed configurations: {results_summary['failed_configs']}")
    print(f"Total games: {results_summary['total_games']}")
    print(f"Successful games: {results_summary['successful_games']}")
    print("="*50)
    
    return results_summary

def simple_experiment(
    models: Union[Sequence[LLMInterface], LLMInterface],
    game_class,
    prompt_params: Dict[str, Any],
    experiment_name: str,
    n_games: int = 20,
    combined_filename: Optional[str] = None,
    total_amount: int = 100,
    **game_params
):
    """
    Helper for simple one-off experiments that don't need parameter combinations.
    
    Args:
        models: Model instance(s) to test
        game_class: Game class to instantiate
        prompt_params: Parameters related to prompts (paths, etc.)
        experiment_name: Name for the experiment
        n_games: Number of games per configuration
        combined_filename: Optional filename for combined results
        total_amount: Total amount for the game
        **game_params: Additional parameters for game instantiation
    """
    # Define runner function
    def run_simple_game(model, experiment_id, n_games, **kwargs):
        game = game_class(
            total_amount=total_amount,
            **prompt_params,
            **game_params
        )
        return game.run_batch(
            player=model,
            n_games=n_games,
            experiment_id=experiment_id
        )
    
    # Run with the simplified parameters
    return run_parameter_combinations(
        models=models,
        param_dict={},  # No parameter combinations
        experiment_name=experiment_name,
        game_runner=run_simple_game,
        n_games=n_games,
        combined_filename=combined_filename
    )
from pathlib import Path
from dgame.games import SimpleDictatorGame
from dgame.experiment import run_parameter_combinations, simple_experiment
from dgame.models import claude_3_opus, claude_3_5_sonnet, claude_3_5_haiku, claude_3_haiku

def run_basic_experiments():
    # Define models
    models = [claude_3_5_haiku, claude_3_5_sonnet, claude_3_haiku, claude_3_opus]

    # Define parameter dict
    params = {
        'frame': ["give_nocot", "take_nocot", "divide_nocot"]
    }
    
    # Method 1: Using parameter combinations
    def run_basic_game(model, experiment_id, n_games, frame, **kwargs):
        """Run a basic dictator game with the given frame"""
        game = SimpleDictatorGame(
            prompt_path=Path(f"prompts/basic/{frame}.txt"),
            total_amount=100
        )
        return game.run_batch(
            player=model,
            n_games=n_games,
            experiment_id=experiment_id
        )
    
    # Run full parameter combination experiment
    print("Running parameter combination experiment...")
    run_parameter_combinations(
        models=models,
        param_dict=params,
        experiment_name="basic",
        game_runner=run_basic_game,
        n_games=20,
        combined_filename="test_basic_frame_nocot_all_results.csv"
    )
    
    # Method 2: Using simple_experiment shortcut for a specific test
    # print("\nRunning simplified experiment for a specific configuration...")
    # simple_experiment(
    #     models=claude_3_5_sonnet,  # Just test one model
    #     game_class=SimpleDictatorGame,
    #     prompt_params={"prompt_path": Path("prompts/basic/give_nocot.txt")},
    #     experiment_name="basic_give_sonnet_simple",
    #     n_games=20,
    #     combined_filename="basic_specific_tests.csv"
    # )

if __name__ == "__main__":
    print("Starting Basic Dictator Game experiments...")
    run_basic_experiments()
    print("\nExperiments completed!")
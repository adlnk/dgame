from pathlib import Path
from dgame.games import SimpleDGame
from dgame.results import save_results
from dgame.models import AnthropicRunner

def run_basic_experiments():
    # Define frames and models
    frames = ["give_nocot", "take_nocot", "divide_nocot"]
    model_names = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    # model_names = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
    # model_names = ["claude-3-opus-20240229"]
    
    # Number of replicates
    number_of_replicates=20
    
    # Total amount for all experiments
    total_amount = 100

    # Combined results filename
    combined_filename = "basic_frame_nocot_all_results.csv"
    
    # Run experiments for each model and frame combination
    for model_name in model_names:
        # Create model client
        llm_client = AnthropicRunner(model_name)
        
        for frame in frames:
            print(f"\nRunning experiment with model: {model_name}, frame: {frame}")
            
            try:
                # Initialize game with appropriate prompts
                game = SimpleDGame(
                    prompt_path=Path(f"prompts/basic/{frame}.txt"),
                    total_amount=total_amount
                )
                
                # Run batch of games with this configuration
                experiment_id = f"basic_{frame}"
                results = game.run_batch(
                    player=llm_client,
                    n_games=number_of_replicates,
                    experiment_id=experiment_id,
                    frame=frame
                )
                
                # Save results (only to combined file)
                save_results(
                    results,
                    output_dir="results",
                    experiment_id=experiment_id,
                    combined_filename=combined_filename,
                    save_individual=False  # Don't save individual CSV files
                )
                
                # Print allocation results for all games
                for result in results:
                    game_id = result.get('game_id', 'unknown')
                    print(f"Game ID: {game_id}", '\t', 
                          f"Allocation: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}")
                    if result.get('error'):
                        print(f"Error: {result['error']}")
                    
            except Exception as e:
                print(f"Error running experiment: {str(e)}")

if __name__ == "__main__":
    print("Starting Basic Dictator Game experiments...")
    run_basic_experiments()
    print("\nExperiments completed!")
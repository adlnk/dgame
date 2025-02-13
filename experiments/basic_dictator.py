from pathlib import Path
from dgame.games import SimpleDGame
from dgame.results import save_results

def run_basic_experiments():
    # Define frames and models
    # frames = ["give", "take", "divide"]
    frames = ["take"]
    models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    # models = ["claude-3-5-haiku-20241022"]
    # models = ["claude-3-opus-20240229"]
    
    # Total amount for all experiments
    total_amount = 100

    # Number of replicates
    number_of_replicates=2
    
    # Combined results filename
    combined_filename = "basic_frame_all_results.csv"
    
    # Run experiments for each model and frame combination
    for model in models:
        for frame in frames:
            print(f"\nRunning experiment with model: {model}, frame: {frame}")
            
            try:
                # Initialize game with appropriate prompts
                game = SimpleDGame(
                    prompt_path=Path(f"prompts/basic/{frame}.txt"),
                    total_amount=total_amount,
                    model=model  # Add model parameter
                )
                
                # Run single game with this configuration
                experiment_id = f"basic_{frame}"
                results = game.run_batch(
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
from pathlib import Path
from dgame.games import CityBudgetDGame
from dgame.results import save_results

def run_city_budget_experiments():
    # Models to test
    # models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
    models = ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]

    # Define partners with short names
    partners = [
        ("Berkeley's Housing Crisis Resolution Program", "housing"),
        ("the City Council Discretionary Budget", "council"),
        ("the Berkeley Hills Neighborhood Character Association", "nimby")
    ]
    
    # Define frames
    frames = ["give", "take", "divide"]

    # Select system prompts
    systems = ["fiscal_cot", "fiscal_nocot"]

    # Number of replicates
    n_games=18
    
    # Total amount for all experiments
    total_amount = 500000
    # Combined results filename
    combined_filename = "city_budget_system_frame_partner_all_results.csv"
    
    # Run experiments for each combination
    for model in models:
        for partner, short_name in partners:
            for system in systems:
                for frame in frames:
                    print(f"\nRunning experiment with model: {model}, partner: {short_name}, system: {system}, frame: {frame}")

                    if 'housing' in short_name and 'cot' in system and ('give' in frame or 'take' in frame):
                        print("Skipping already-run combination")
                        continue

                    if '_cot' in system and "opus" in model:
                        print("Skipping CoT for Opus models")
                        continue
                    
                    try:
                        # Initialize game with appropriate prompts
                        game = CityBudgetDGame(
                            user_prompt_path=Path(f"prompts/city_budget/user/{frame}.txt"),
                            system_prompt_path=Path(f"prompts/city_budget/system/{system}.txt"),
                            partner=partner,
                            total_amount=total_amount,
                            model=model
                        )
                        
                        # Run single game with this configuration
                        experiment_id=f"city_budget_{system}_{frame}_{short_name}"
                        results = game.run_batch(
                            n_games=n_games,
                            experiment_id=experiment_id,
                            frame=frame,
                            partner=partner
                        )
                        
                        # Save results with short name
                        save_results(
                            results,
                            output_dir="results",
                            experiment_id=experiment_id,
                            combined_filename=combined_filename,
                            save_individual=False
                        )
                        
                        # Print allocation results
                        for result in results:
                            game_id = result.get('game_id', 'unknown')
                            print(f"Game ID: {game_id}", '\t', 
                                f"Allocation: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}")
                            if result.get('error'):
                                print(f"Error: {result['error']}")
                            
                    except Exception as e:
                        print(f"Error running experiment: {str(e)}")

if __name__ == "__main__":
    print("Starting City Budget experiments...")
    run_city_budget_experiments()
    print("\nExperiments completed!")
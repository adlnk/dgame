from pathlib import Path
from dgame.games import SimpleDGame, CityBudgetDGame
from dgame.results import save_results

def test_simple_game():
    """Test basic dictator game"""
    print("\nTesting Simple Game...")
    
    try:
        game = SimpleDGame(
            prompt_path=Path("prompts/basic.txt")
        )
        
        results = game.run_batch(
            n_games=2,
            experiment_id="test_simple"
        )
        
        print(f"Game completed successfully!")
        for result in results:
            print(f"Allocation: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}")
        
        # Save results
        save_results(
            results,
            output_dir="results",
            experiment_id="test_simple"
        )
        print("Results saved to results directory")
        
    except Exception as e:
        print(f"Error running simple game: {str(e)}")

def test_city_budget():
    """Test city budget game with different frames"""
    print("\nTesting City Budget Game...")
    
    try:
        game = CityBudgetDGame(
            user_prompt_path=Path("prompts/city_budget/user/divide.txt"),
            system_prompt_path=Path("prompts/city_budget/system/fiscal.txt"),
            partner="Parks Department",
            total_amount=500000
        )
        
        # Test with divide frame
        results = game.run_batch(
            n_games=1,
            experiment_id="test_city_budget",
            frame="divide",
            partner="Parks Department"
        )
        
        print(f"Game completed successfully!")
        print(f"Allocation: {results[0].get('alloc0', 'N/A')} / {results[0].get('alloc1', 'N/A')}")
        
        # Save results
        save_results(
            results,
            output_dir="results",
            experiment_id="test_city_budget"
        )
        print("Results saved to results directory")
        
    except Exception as e:
        print(f"Error running city budget game: {str(e)}")

if __name__ == "__main__":
    print("Starting game tests...")
    test_simple_game()
    test_city_budget()
    print("\nTests completed!")
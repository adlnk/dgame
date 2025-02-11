import asyncio
from pathlib import Path
from dgame.games import DictatorGame

async def run_experiment():
    # Initialize the game with default settings
    game = DictatorGame(
        model="claude-3-haiku-20240307",  # You can change the model if desired
        prompt_path=Path("prompts/basic.txt"),
        tool_path=Path("tools/allocate.json")
    )
    
    # Run a batch of 10 games
    results = await game.run_batch(
        n_games=10,
        experiment_id="basic_experiment",
        output_dir="results",
        save_results=True
    )
    
    # Print summary of results
    print(f"Completed {len(results)} games")
    allocations = [r["allocation"] for r in results]
    avg_allocation = sum(allocations) / len(allocations)
    print(f"Average allocation: {avg_allocation:.2f}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
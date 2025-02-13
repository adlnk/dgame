from pathlib import Path
import pandas as pd
from typing import Dict, List, Union, Tuple

def save_results(
    results: List[Dict],
    output_dir: Union[str, Path],
    experiment_id: str = None,
    batch_id: str = None,
    combined_filename: str = None,
    save_individual: bool = True
) -> Tuple[Path, Path]:
    """
    Save game results to CSV file and responses to separate text files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to store results
        experiment_id: Optional identifier for experiment
        batch_id: Optional identifier for batch
        combined_filename: Optional filename for appending to a combined CSV
        save_individual: Whether to save individual experiment CSV files
        
    Returns:
        Tuple of (results_path, responses_dir) where results_path will be None if only saving combined
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create responses directory
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(exist_ok=True)
    
    # Save responses to separate files
    for result in results:
        if "response" in result:
            response_path = responses_dir / f"{result['game_id']}.txt"
            response_path.write_text(result["response"])
            # Remove response from result dict to keep CSV clean
            del result["response"]
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    output_path = None
    if save_individual:
        # Save individual experiment results
        parts = []
        if experiment_id:
            parts.append(experiment_id)
        if batch_id:
            parts.append(batch_id)
        # Add model name from first result (all results in batch use same model)
        if results and "model" in results[0]:
            parts.append(results[0]["model"])
        parts.append("results.csv")
        filename = "_".join(parts)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
    
    # Save/append to combined results if specified
    if combined_filename:
        combined_path = output_dir / combined_filename
        if combined_path.exists():
            df.to_csv(combined_path, mode='a', header=False, index=False)
        else:
            df.to_csv(combined_path, index=False)
        if not save_individual:
            output_path = combined_path
    
    return output_path, responses_dir

def load_results(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        DataFrame containing results
    """
    return pd.read_csv(Path(filepath))
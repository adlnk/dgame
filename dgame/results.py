from pathlib import Path
import pandas as pd
from typing import Dict, List, Union, Tuple

def save_results(
    results: List[Dict],
    output_dir: Union[str, Path],
    experiment_id: str = None,
    batch_id: str = None
) -> Tuple[Path, Path]:
    """
    Save game results to CSV file and responses to separate text files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to store results
        experiment_id: Optional identifier for experiment
        batch_id: Optional identifier for batch
        
    Returns:
        Tuple of (results_path, responses_dir)
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
    
    # Construct filename
    parts = []
    if experiment_id:
        parts.append(experiment_id)
    if batch_id:
        parts.append(batch_id)
    parts.append("results.csv")
    filename = "_".join(parts)
    
    # Save
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    
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
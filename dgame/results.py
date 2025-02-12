from pathlib import Path
import pandas as pd
from typing import Dict, List, Union

def save_results(
    results: List[Dict],
    output_dir: Union[str, Path],
    experiment_id: str = None,
    batch_id: str = None
) -> Path:
    """
    Save game results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to store results
        experiment_id: Optional identifier for experiment
        batch_id: Optional identifier for batch
        
    Returns:
        Path to saved results file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
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
    
    return output_path

def load_results(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        DataFrame containing results
    """
    return pd.read_csv(Path(filepath))
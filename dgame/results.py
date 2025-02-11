from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union

def save_results(
    results: List[Dict],
    output_dir: Union[str, Path] = "results",
    experiment_id: str = None,
    batch_id: str = None
) -> Path:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to store results (default: "results")
        experiment_id: Optional identifier for experiment
        batch_id: Optional identifier for batch
        
    Returns:
        Path to saved results file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Add metadata to each result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for result in results:
        result["timestamp"] = timestamp
        if experiment_id:
            result["experiment_id"] = experiment_id
        if batch_id:
            result["batch_id"] = batch_id
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    filename = f"results_{timestamp}.csv"
    if experiment_id:
        filename = f"{experiment_id}_{filename}"
    
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
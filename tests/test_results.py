# tests/test_results.py
import pytest
import pandas as pd
import os

from dgame.results import save_results, load_results


class TestResultsHandling:
    """Tests for results handling functions."""
    
    def test_save_results_individual(self, sample_results, temp_dir):
        """Test saving individual result files."""
        # Call function
        result = save_results(
            results=sample_results,
            output_dir=temp_dir,
            experiment_id="test_experiment",
            batch_id="test_batch",
            save_individual=True
        )
        
        # Check result
        assert result["results_path"].exists()
        assert result["responses_dir"].exists()
        
        # Check that files were created
        assert (temp_dir / "test_experiment_test_batch_test-model_results.csv").exists()
        assert (temp_dir / "responses" / "test-game-1.txt").exists()
        assert (temp_dir / "responses" / "test-game-2.txt").exists()
        
        # Check content of saved CSV
        df = pd.read_csv(result["results_path"])
        assert len(df) == 2
        assert "response" not in df.columns  # Responses should be saved separately
        assert df.iloc[0]["game_id"] == "test-game-1"
        assert df.iloc[0]["alloc0"] == 60
        assert df.iloc[0]["alloc1"] == 40
    
    def test_save_results_combined(self, sample_results, temp_dir):
        """Test saving to a combined result file."""
        # Call function
        result1 = save_results(
            results=sample_results[:1],
            output_dir=temp_dir,
            experiment_id="test_experiment_1",
            combined_filename="combined.csv",
            save_individual=False
        )
        
        result2 = save_results(
            results=sample_results[1:],
            output_dir=temp_dir,
            experiment_id="test_experiment_2",
            combined_filename="combined.csv",
            save_individual=False
        )
        
        # Check result
        combined_path = temp_dir / "combined.csv"
        assert combined_path.exists()
        
        # Check content of combined CSV
        df = pd.read_csv(combined_path)
        assert len(df) == 2
        assert df.iloc[0]["game_id"] == "test-game-1"
        assert df.iloc[1]["game_id"] == "test-game-2"
    
    def test_load_results(self, sample_results, temp_dir):
        """Test loading results from CSV."""
        # Save results first
        save_results(
            results=sample_results,
            output_dir=temp_dir,
            experiment_id="test_experiment",
            save_individual=True
        )
        
        # Get path to saved CSV
        csv_path = next(temp_dir.glob("*.csv"))
        
        # Load results
        df = load_results(csv_path)
        
        # Check loaded data
        assert len(df) == 2
        assert df.iloc[0]["game_id"] == "test-game-1"
        assert df.iloc[0]["alloc0"] == 60
        assert df.iloc[0]["alloc1"] == 40

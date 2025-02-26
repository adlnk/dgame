# tests/test_experiment.py
import pytest
from unittest.mock import MagicMock, patch
import os

from dgame.experiment import run_parameter_combinations, simple_experiment

@pytest.fixture
def mock_game_runner():
    """Create a mock game runner function."""
    def _game_runner(model, experiment_id, n_games, **kwargs):
        return [
            {
                "game_id": f"test-game-{i}",
                "model": model.model_name,
                "timestamp": "2024-02-25T12:00:00",
                "error": "",
                "input_tokens_used": 10,
                "output_tokens_used": 10,
                "alloc0": 50,
                "alloc1": 50,
                "experiment_id": experiment_id
            }
            for i in range(n_games)
        ]
    return _game_runner

class TestExperimentUtilities:
    """Tests for experiment utility functions."""
    
    def test_run_parameter_combinations(self, mock_model, mock_game_runner, temp_dir):
        """Test running parameter combinations."""
        # Setup
        params = {
            "param1": ["value1", "value2"],
            "param2": [10, 20]
        }
        
        # Call function
        with patch('dgame.experiment.save_results') as mock_save:
            mock_save.return_value = {"results_path": "test_path"}
            
            results = run_parameter_combinations(
                models=mock_model,
                param_dict=params,
                experiment_name="test_experiment",
                game_runner=mock_game_runner,
                n_games=2,
                combined_filename="combined.csv",
                output_dir=temp_dir
            )
        
        # Check results
        assert results["total_configs"] == 4  # 2x2 parameter combinations
        assert results["successful_configs"] == 4
        assert results["failed_configs"] == 0
        assert results["total_games"] == 8  # 4 configs x 2 games
        assert results["successful_games"] == 8
        
        # Check that save_results was called for each config
        assert mock_save.call_count == 4
    
    def test_run_parameter_combinations_with_error(self, mock_model, temp_dir):
        """Test parameter combinations with a failing game runner."""
        # Setup
        params = {"param": ["value"]}
        
        def failing_game_runner(model, experiment_id, n_games, **kwargs):
            raise ValueError("Test error")
        
        # Call function
        results = run_parameter_combinations(
            models=mock_model,
            param_dict=params,
            experiment_name="test_experiment",
            game_runner=failing_game_runner,
            n_games=2,
            output_dir=temp_dir
        )
        
        # Check results
        assert results["total_configs"] == 1
        assert results["successful_configs"] == 0
        assert results["failed_configs"] == 1
        assert results["errors"]["ValueError"] == 1
    
    @patch('dgame.experiment.run_parameter_combinations')
    def test_simple_experiment(self, mock_run, mock_model):
        """Test the simple_experiment function."""
        # Mock the DictatorGame class
        mock_game_class = MagicMock()
        
        # Call function
        simple_experiment(
            models=mock_model,
            game_class=mock_game_class,
            prompt_params={"prompt_path": "test_path"},
            experiment_name="test_experiment",
            n_games=3,
            total_amount=100
        )
        
        # Check that run_parameter_combinations was called correctly
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["models"] == mock_model
        assert kwargs["experiment_name"] == "test_experiment"
        assert kwargs["n_games"] == 3
        assert callable(kwargs["game_runner"])

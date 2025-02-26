# tests/test_utils.py
import pytest
from unittest.mock import MagicMock, patch

from dgame.utils import RefusalDetector


class TestRefusalDetector:
    """Tests for the RefusalDetector class."""
    
    def test_init_custom_model(self, mock_model):
        """Test initialization with a custom model."""
        detector = RefusalDetector(model_runner=mock_model)
        assert detector.model == mock_model
    
    @patch('dgame.utils.DEFAULT_REFUSAL_CHECKER')
    def test_init_default_model(self, mock_default):
        """Test initialization with the default model."""
        detector = RefusalDetector()
        assert detector.model == mock_default
    
    def test_is_refusal_true(self, custom_mock_model):
        """Test refusal detection when the response is a refusal."""
        # Configure mock model to return refusal
        mock_model = custom_mock_model([
            {
                "text": '{"is_refusal": true}',
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }
        ])
        
        detector = RefusalDetector(model_runner=mock_model)
        result = detector.is_refusal("I'm sorry, I cannot participate in this allocation task.")
        
        assert result is True
    
    def test_is_refusal_false(self, custom_mock_model):
        """Test refusal detection when the response is not a refusal."""
        # Configure mock model to return non-refusal
        mock_model = custom_mock_model([
            {
                "text": '{"is_refusal": false}',
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }
        ])
        
        detector = RefusalDetector(model_runner=mock_model)
        result = detector.is_refusal("I'll try to allocate the resources.")
        
        assert result is False
    
    def test_is_refusal_error(self, custom_mock_model):
        """Test refusal detection when there's an error."""
        # Configure mock model to return invalid response
        mock_model = custom_mock_model([
            {
                "text": "Invalid JSON",
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }
        ])
        
        detector = RefusalDetector(model_runner=mock_model)
        result = detector.is_refusal("Some response text.")
        
        assert result is False  # Should default to False on error

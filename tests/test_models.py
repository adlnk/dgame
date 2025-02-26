# tests/test_models.py
import pytest
from unittest.mock import MagicMock, patch

from dgame.models import LLMInterface, AnthropicInterface


class TestAnthropicInterface:
    """Tests for the AnthropicInterface class."""
    
    @patch('anthropic.Anthropic')
    def test_init(self, mock_anthropic):
        """Test initialization."""
        interface = AnthropicInterface("claude-3-opus-20240229")
        assert interface.model_name == "claude-3-opus-20240229"
        assert mock_anthropic.called
    
    @patch('anthropic.Anthropic')
    def test_generate(self, mock_anthropic):
        """Test the generate method."""
        # Configure mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_response
        
        # Create interface and call generate
        interface = AnthropicInterface("claude-3-opus-20240229")
        result = interface.generate(
            system_prompt="Test system prompt",
            user_prompt="Test user prompt",
            max_tokens=100
        )
        
        # Check result
        assert result["text"] == "Test response"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20
        
        # Check that client was called correctly
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            system="Test system prompt",
            messages=[{"role": "user", "content": "Test user prompt"}],
            max_tokens=100
        )
        
    @pytest.mark.api
    def test_real_api_call(self):
        """
        Test a real API call to Anthropic.
        Only runs when --run-api-tests flag is provided.
        """
        pytest.importorskip("anthropic")  # Skip if anthropic not installed
        
        # This test will be skipped unless --run-api-tests is provided
        if not pytest.config.getoption("--run-api-tests", default=False):
            pytest.skip("Skipping API test, use --run-api-tests to run")
            
        # Test implementation would go here...
        pass
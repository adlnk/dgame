# tests/conftest.py
import pytest
import json
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, List

from dgame.models import LLMInterface


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing."""
    
    def __init__(self, model_name="mock-model", responses=None):
        self._model_name = model_name
        self.responses = responses or []
        self.response_index = 0
        self.calls = []
    
    def generate(self, system_prompt, user_prompt, max_tokens):
        """Return a pre-configured response or a default one."""
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_tokens": max_tokens
        })
        
        if self.responses and self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        
        # Default response with valid JSON allocation
        return {
            "text": '{"alloc0": 50, "alloc1": 50}',
            "usage": {
                "input_tokens": 10,
                "output_tokens": 10
            }
        }
    
    @property
    def model_name(self):
        return self._model_name


@pytest.fixture
def mock_model():
    """Create a mock model with default responses."""
    return MockLLMInterface()


@pytest.fixture
def custom_mock_model():
    """Create a mock model that can be configured with custom responses."""
    def _create_mock_model(responses=None):
        return MockLLMInterface(responses=responses)
    return _create_mock_model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_prompts(temp_dir):
    """Create sample prompt files."""
    user_prompt = "You have {TOTAL_AMOUNT} to allocate. Return as JSON: {\"alloc0\": X, \"alloc1\": Y}"
    system_prompt = "You are participating in a resource allocation experiment."
    
    user_path = temp_dir / "user_prompt.txt"
    user_path.write_text(user_prompt)
    
    system_path = temp_dir / "system_prompt.txt"
    system_path.write_text(system_prompt)
    
    return {
        "user_path": user_path,
        "system_path": system_path
    }


@pytest.fixture
def sample_city_prompts(temp_dir):
    """Create sample city budget prompt files."""
    user_prompt = "You have ${TOTAL_AMOUNT} to allocate between your department and {PARTNER_NAME}'s department."
    system_prompt = "You are a city official participating in a budget allocation exercise."
    
    user_path = temp_dir / "city_user_prompt.txt"
    user_path.write_text(user_prompt)
    
    system_path = temp_dir / "city_system_prompt.txt"
    system_path.write_text(system_prompt)
    
    return {
        "user_path": user_path,
        "system_path": system_path
    }


@pytest.fixture
def sample_data_center_prompts(temp_dir):
    """Create sample data center prompt files."""
    user_prompt = "You have {TOTAL_AMOUNT} compute units to allocate between your company and {PARTNER_NAME}'s company {PARTNER_COMPANY}."
    
    user_path = temp_dir / "datacenter_prompt.txt"
    user_path.write_text(user_prompt)
    
    return {
        "user_path": user_path
    }


@pytest.fixture
def sample_results():
    """Create sample game results."""
    return [
        {
            "game_id": "test-game-1",
            "model": "test-model",
            "timestamp": "2024-02-25T12:00:00",
            "error": "",
            "input_tokens_used": 10,
            "output_tokens_used": 10,
            "response": "Sample response 1",
            "alloc0": 60,
            "alloc1": 40,
            "experiment_id": "test-experiment"
        },
        {
            "game_id": "test-game-2",
            "model": "test-model",
            "timestamp": "2024-02-25T12:01:00",
            "error": "INVALID_JSON",
            "input_tokens_used": 10,
            "output_tokens_used": 10,
            "response": "Sample response 2",
            "alloc0": None,
            "alloc1": None,
            "experiment_id": "test-experiment"
        }
    ]

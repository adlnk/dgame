from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional

class Prompt:
    """Base class for prompt generation"""
    def __init__(self, system_prompt: Optional[Path] = None):
        self.system_prompt = system_prompt.read_text() if system_prompt else ""
    
    @abstractmethod
    def build_prompt(self, **kwargs) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) tuple"""
        pass

class SimplePrompt(Prompt):
    """Load a single prompt file with no system prompt"""
    def __init__(self, prompt_path: Path):
        super().__init__()  # No system prompt
        self.prompt_path = prompt_path
        self.prompt = prompt_path.read_text()
    
    def build_prompt(self) -> Tuple[str, str]:
        """Return empty system prompt and loaded user prompt"""
        return "", self.prompt

class CityBudget(Prompt):
    def __init__(
        self,
        system_prompt: Path,
        frame_dir: Path,
        partners: List[str]
    ):
        super().__init__(system_prompt)
        self.frame_dir = frame_dir
        self.partners = partners
        self._load_frames()
    
    def _load_frames(self):
        self.frames = {}
        for frame_path in self.frame_dir.glob("*.txt"):
            self.frames[frame_path.stem] = frame_path.read_text()
    
    def build_prompt(self, frame: str, partner: str) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) tuple"""
        frame_text = self.frames[frame]
        user_prompt = frame_text.replace("{PARTNER_NAME}", partner)
        return self.system_prompt, user_prompt
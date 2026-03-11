import os
import subprocess
import uuid
import logging
from typing import Optional

logger = logging.getLogger("axiom_vox.piper")

class PiperEngine:
    """
    Lightweight wrapper for Piper TTS.
    Designed for ultra-fast, high-quality CPU-based synthesis.
    """

    def __init__(self, model_path: Optional[str] = None):
        # Default to a high-quality voice in the container
        self.model_path = model_path or "/app/models/en_US-lessac-high.onnx"
        self.config_path = self.model_path + ".json"

    def generate_to_file(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to a WAV file using the piper CLI.
        """
        try:
            # Piper CLI: echo "text" | piper --model model.onnx --output_file output.wav
            cmd = [
                "piper",
                "--model", self.model_path,
                "--config", self.config_path,
                "--output_file", output_path
            ]
            
            logger.info(f"[PIPER] Synthesizing: {text[:50]}...")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0:
                logger.info(f"[PIPER] Success -> {output_path}")
                return True
            else:
                logger.error(f"[PIPER] Error: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[PIPER] Failed to generate: {e}")
            return False

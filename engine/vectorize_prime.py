#!/usr/bin/env python3
"""
PRIME Voice Distiller (Corrected)
--------------------------------
"Distills" the reference audio into a concentrated Conditionals object (.pt).
Reduces synthesis latency by ~300ms and locks the vocal fingerprint.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axiom_vox.chatterbox_engine import ChatterboxEngine, PRIME_REF_DEFAULT, PRIME_VECTOR_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("distiller")

def main():
    print("\n" + "="*50)
    print("   AXIØM PRIME: VOICE DISTILLERY")
    print("="*50 + "\n")

    engine = ChatterboxEngine()
    
    if not os.path.exists(PRIME_REF_DEFAULT):
        logger.error(f"Reference audio not found: {PRIME_REF_DEFAULT}")
        return

    logger.info(f"Source: {PRIME_REF_DEFAULT}")
    logger.info("Initializing neural engine for distillation...")
    
    if not engine.load():
        logger.error("Failed to load synthesis engine.")
        return

    logger.info("Extracting vocal essence (Conditionals)...")
    t0 = time.time()
    
    try:
        # Vectorize method now correctly uses prepare_conditionals and model.conds
        vector = engine.vectorize(PRIME_REF_DEFAULT, PRIME_VECTOR_DEFAULT)
        
        if vector is not None:
            duration = time.time() - t0
            logger.info(f"SUCCESS: Distillate saved to {PRIME_VECTOR_DEFAULT}")
            logger.info(f"Process complete in {duration:.2f}s")
        else:
            logger.error("Vectorization method returned None.")
            
    except Exception as e:
        logger.error(f"Distillation failed: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PRIME Voice Demo
================

Hear PRIME speak. Uses the large Qwen3-TTS-1.7B model with PRIME's
locked voice identity and context-aware speaking modes.

Usage:
    python -m axiom_vox.prime_voice_demo

Requirements:
    pip3 install qwen-tts torch sounddevice soundfile
"""

import sys
import time

# Check dependencies before importing anything heavy
def check_deps():
    missing = []
    for mod, pkg in [("torch", "torch"), ("qwen_tts", "qwen-tts"),
                     ("sounddevice", "sounddevice"), ("soundfile", "soundfile")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Install: pip3 install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import numpy as np
import sounddevice as sd
import torch

from axiom_vox.prime_voice import (
    get_prime_identity,
    get_identity_manager,
    detect_speaking_mode,
    SpeakingModeType,
    PRIME_SPEAKING_MODES,
)
from axiom_vox.synthesis import VoxSynthesizer, VoiceConfig, AudioFormat


def print_header():
    print()
    print("=" * 60)
    print("  PRIME VOICE - Sovereign Voice of AXIOM")
    print("=" * 60)
    print()


def print_identity():
    identity = get_prime_identity()
    dna = identity.vocal_dna
    vec = identity.voice_vector

    print(f"  Identity:    {identity.identity_name} ({identity.identity_id})")
    print(f"  Voice ID:    {identity.vox_voice_id}")
    print(f"  Lock Level:  {identity.lock_level.value}")
    print(f"  Model:       Qwen3-TTS-1.7B (large)")
    print()
    print(f"  Voice Vector: formality={vec.formality:+.1f} "
          f"temp={vec.temperature:+.1f} energy={vec.energy:+.1f} "
          f"authority={vec.authority:+.1f}")
    print(f"                certainty={vec.certainty:+.1f} "
          f"intimacy={vec.intimacy:+.1f} "
          f"abstraction={vec.abstraction:+.1f} "
          f"complexity={vec.complexity:+.1f}")
    print()
    print(f"  Vocal DNA:   pitch={dna.target_pitch_hz:.0f}Hz "
          f"rate={dna.target_speaking_rate:.2f}x "
          f"authority={dna.target_authority:.0%} "
          f"trust={dna.target_trust:.0%}")
    print()


def init_synthesizer():
    """Initialize the large Qwen3-TTS model."""
    print("  Loading Qwen3-TTS-1.7B...")
    device = None
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"  Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"  Device: CUDA GPU")
    else:
        device = "cpu"
        print(f"  Device: CPU (slower)")

    start = time.time()
    synth = VoxSynthesizer(model_size="large", device=device)
    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    print()
    return synth


def speak(synth, text, mode=None, label=None):
    """Synthesize and play PRIME speaking."""
    # Auto-detect mode if not specified
    if mode is None:
        mode = detect_speaking_mode(text)

    profile = PRIME_SPEAKING_MODES[mode]
    identity = get_prime_identity()
    dna = identity.vocal_dna

    # Build voice config with PRIME's DNA + mode adjustments
    voice = VoiceConfig(
        voice_id=identity.vox_voice_id,
        speaking_rate=dna.target_speaking_rate * profile.rate_multiplier,
        pitch=profile.pitch_shift,
        emotion=identity.emotion_preset,
    )

    if label:
        print(f"  [{label}]")
    print(f"  Mode: {mode.value.upper()}")
    print(f"  Rate: {voice.speaking_rate:.2f}x | Pitch: {voice.pitch:+.1f}")
    print(f"  \"{text}\"")
    print()

    # Synthesize
    start = time.time()
    result = synth.synthesize(text=text, voice=voice, output_format=AudioFormat.WAV)
    elapsed = time.time() - start

    if result.success and result.audio_data:
        audio = np.frombuffer(result.audio_data, dtype=np.float32)
        duration = len(audio) / result.sample_rate

        print(f"  Synthesized: {duration:.1f}s audio in {elapsed:.1f}s "
              f"(RTF: {elapsed/duration:.2f}x)")
        print(f"  Playing...")
        print()

        sd.play(audio, samplerate=result.sample_rate)
        sd.wait()
        return True
    else:
        print(f"  Synthesis failed: {result.error}")
        print()
        return False


def run_demo():
    """Run the full PRIME voice demo."""
    print_header()
    print_identity()

    print("-" * 60)
    print("  Initializing...")
    print()

    synth = init_synthesizer()

    print("=" * 60)
    print("  PRIME is ready to speak.")
    print("=" * 60)
    print()

    # Demo utterances across different modes
    demos = [
        # (text, mode_override, label)
        (
            "Good morning. All systems nominal. PRIME Voice is online.",
            SpeakingModeType.CONVERSATIONAL,
            "GREETING",
        ),
        (
            "System status report. 47 services operational. "
            "Latency at 12 milliseconds p99. Memory utilization at 62 percent. "
            "All health checks passing.",
            SpeakingModeType.BRIEFING,
            "BRIEFING",
        ),
        (
            "Warning. Database connection pool at 95 percent capacity. "
            "Initiating failover protocol.",
            SpeakingModeType.ALERT,
            "ALERT",
        ),
        (
            "The analysis reveals three patterns. First, request volume peaks "
            "correlate with cache eviction. Second, the memory curve suggests "
            "a slow leak in the session handler. Third, the error rate is "
            "concentrated in a single service.",
            SpeakingModeType.REFLECTIVE,
            "ANALYSIS",
        ),
        (
            "Execute deployment sequence. Push version 2.1.0 to production cluster. "
            "Enable blue-green routing. Monitor for 5 minutes before full cutover.",
            SpeakingModeType.DIRECTIVE,
            "COMMAND",
        ),
        (
            "VØX version 1.3.0 is now live. PRIME Voice is operational. "
            "The AXIOM system can now speak.",
            SpeakingModeType.CEREMONIAL,
            "ANNOUNCEMENT",
        ),
    ]

    for text, mode, label in demos:
        print("-" * 60)
        speak(synth, text, mode=mode, label=label)
        time.sleep(0.5)  # Brief pause between demos

    print("=" * 60)
    print("  Demo complete. PRIME Voice is operational.")
    print("=" * 60)
    print()

    # Interactive mode
    print("  Enter text for PRIME to speak (or 'quit' to exit):")
    print("  Prefix with mode: [alert] [briefing] [reflective] [directive]")
    print()

    while True:
        try:
            user_input = input("  PRIME> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # Check for mode prefix
        mode_override = None
        for mt in SpeakingModeType:
            prefix = f"[{mt.value}]"
            if user_input.lower().startswith(prefix):
                mode_override = mt
                user_input = user_input[len(prefix):].strip()
                break

        print()
        speak(synth, user_input, mode=mode_override)

    print("  PRIME Voice signing off.")
    print()


if __name__ == "__main__":
    run_demo()

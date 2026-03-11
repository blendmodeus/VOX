import torch
import os
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Absolute path to your reference wav
REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")

if not os.path.exists(REF_WAV):
    print(f"ERROR: No reference file at {REF_WAV}")
    exit(1)

print("Loading model for recon...")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Capture state BEFORE
state_0 = set(model.__dict__.keys())

print(f"Distilling essence from {REF_WAV}...")
result = model.prepare_conditionals(REF_WAV)

# Capture state AFTER
state_1 = set(model.__dict__.keys())

new_keys = state_1 - state_0
print("\n" + "="*50)
print("   VOICE LATENT RECONNAISSANCE")
print("="*50)
print(f"Method returned type: {type(result)}")
print(f"New attributes found: {new_keys}")

# Try to find which one holds the tensor
for key in state_1:
    val = getattr(model, key)
    if isinstance(val, torch.Tensor):
        # We're looking for the speaker latent
        if val.ndim > 1:
            print(f"Potential latent attribute: {key} (shape={val.shape})")

print("="*50)

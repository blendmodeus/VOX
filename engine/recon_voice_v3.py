import torch
import os
import copy
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Capture all attributes
def get_snapshot(obj):
    snapshot = {}
    for m in dir(obj):
        try:
            val = getattr(obj, m)
            if isinstance(val, torch.Tensor):
                snapshot[m] = val.clone()
            else:
                snapshot[m] = copy.copy(val)
        except:
            pass
    return snapshot

print(f"Distilling from {REF_WAV}...")
snap_0 = get_snapshot(model)
model.prepare_conditionals(REF_WAV)
snap_1 = get_snapshot(model)

print("\n" + "="*50)
print("   DEEP VOICE LATENT RECON")
print("="*50)

for k in snap_1:
    v0 = snap_0.get(k)
    v1 = snap_1[k]
    if v0 != v1:
        if isinstance(v1, torch.Tensor):
            if not torch.equal(v0, v1):
                print(f"MODIFIED TENSOR: {k} (shape={v1.shape})")
        else:
             print(f"MODIFIED ATTR: {k} (old={v0}, new={v1})")

# Check if it has a sub-model
if hasattr(model, "model"):
    print("Checking internal model...")
    # recurse if needed or just dump its dict
    # but let's see what top level changes first
    
print("="*50)

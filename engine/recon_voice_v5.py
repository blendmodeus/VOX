import torch
import os
import copy
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")

def get_snapshot(obj):
    snapshot = {}
    try:
        data = vars(obj)
    except:
        return {}
        
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            snapshot[k] = v.detach().cpu().clone()
        elif isinstance(v, (int, float, str, bool, type(None))):
            snapshot[k] = v
        # Skip other types to avoid errors
    return snapshot

print("Loading model for recon v5...")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

snap_0 = get_snapshot(model)

print(f"Distilling from {REF_WAV}...")
model.prepare_conditionals(REF_WAV)

snap_1 = get_snapshot(model)

print("\n" + "="*50)
print("   DEEP VOICE LATENT RECON V5")
print("="*50)

found = False
all_keys = set(snap_0.keys()) | set(snap_1.keys())

for k in all_keys:
    v0 = snap_0.get(k)
    v1 = snap_1.get(k)
    
    if v0 is None and v1 is not None:
        print(f"NEW ATTR: {k} (type={type(v1)})")
        found = True
    elif v1 is None:
        pass
    elif isinstance(v1, torch.Tensor):
        if not isinstance(v0, torch.Tensor) or not torch.equal(v0, v1):
            print(f"MODIFIED TENSOR: {k} (shape={v1.shape})")
            found = True
    else:
        try:
            if v0 != v1:
                print(f"MODIFIED ATTR: {k} (old={v0}, new={v1})")
                found = True
        except:
            pass

if not found:
    print("NO CHANGES detected in top-level attributes.")

print("="*50)

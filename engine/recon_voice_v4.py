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
                snapshot[m] = val.detach().cpu().clone()
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
print("   DEEP VOICE LATENT RECON V4")
print("="*50)

for k in snap_1:
    v0 = snap_0.get(k)
    v1 = snap_1[k]
    
    changed = False
    if isinstance(v1, torch.Tensor):
        if v0 is None or not torch.equal(v0, v1):
            changed = True
            print(f"MODIFIED TENSOR: {k} (shape={v1.shape})")
    else:
        if v0 != v1:
            changed = True
            print(f"MODIFIED ATTR: {k} (old={v0}, new={v1})")

# Special check for internal state names
if not any(k in snap_1 for k in ['_cond', 'cond', 'conditionals', 'speaker_latent']):
     print("\nProbing for hidden tensor attributes...")
     for k in snap_1:
         if isinstance(snap_1[k], torch.Tensor):
             print(f"  - Found tensor: {k} (shape={snap_1[k].shape})")

print("="*50)

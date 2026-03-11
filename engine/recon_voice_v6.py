import torch
import os
import copy
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")

def get_snapshot(obj):
    snapshot = {}
    if obj is None: return {}
    try:
        data = vars(obj)
    except:
        return {}
        
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            snapshot[k] = v.detach().cpu().clone()
        elif isinstance(v, (int, float, str, bool, type(None))):
            snapshot[k] = v
    return snapshot

print("Loading model for recon v6...")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Capture top level AND internal model
snap_top_0 = get_snapshot(model)
snap_int_0 = None
if hasattr(model, 'model'):
    snap_int_0 = get_snapshot(model.model)

print(f"Distilling from {REF_WAV}...")
model.prepare_conditionals(REF_WAV)

snap_top_1 = get_snapshot(model)
snap_int_1 = None
if hasattr(model, 'model'):
    snap_int_1 = get_snapshot(model.model)

print("\n" + "="*50)
print("   DEEP VOICE LATENT RECON V6")
print("="*50)

def compare(s0, s1, label):
    found = False
    all_keys = set(s0.keys()) | set(s1.keys())
    for k in all_keys:
        v0 = s0.get(k)
        v1 = s1.get(k)
        if v1 is None: continue
        if isinstance(v1, torch.Tensor):
            if v0 is None or not torch.equal(v0, v1):
                print(f"[{label}] MODIFIED TENSOR: {k} (shape={v1.shape})")
                found = True
        else:
            try:
                if v0 != v1:
                    print(f"[{label}] MODIFIED ATTR: {k} (old={v0}, new={v1})")
                    found = True
            except:
                pass
    return found

f1 = compare(snap_top_0, snap_top_1, "TOP")
if snap_int_0 and snap_int_1:
    f2 = compare(snap_int_0, snap_int_1, "INTERNAL")
else:
    f2 = False

if not f1 and not f2:
    print("NO CHANGES detected in top or internal attributes.")
    print("Checking for _ conditionals specifically...")
    for m in dir(model):
        if 'cond' in m.lower():
            print(f"Found attribute: {m}")

print("="*50)

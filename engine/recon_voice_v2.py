import torch
import os
import copy
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Clone dict
dict_0 = {k: v.clone() if isinstance(v, torch.Tensor) else None for k, v in model.__dict__.items()}

print(f"Distilling from {REF_WAV}...")
model.prepare_conditionals(REF_WAV)

dict_1 = model.__dict__

print("\n" + "="*50)
print("   VOICE LATENT CONTENT-CHANGE RECON")
print("="*50)

found = False
for k, v1 in dict_1.items():
    if isinstance(v1, torch.Tensor):
        v0 = dict_0.get(k)
        if v0 is not None:
            if not torch.equal(v0, v1):
                print(f"MODIFIED TENSOR: {k} (shape={v1.shape})")
                found = True
        else:
            # Maybe it wasn't a tensor before?
            print(f"NEW TENSOR: {k} (shape={v1.shape})")
            found = True

if not found:
    print("NO MODIFIED TENSORS detected in model.__dict__.")
    print("Checking if attributes were added beyond __dict__ (slots?)...")
    # check everything in dir
    for m in dir(model):
        if not m.startswith("__"):
            attr = getattr(model, m)
            if isinstance(attr, torch.Tensor):
                print(f"Active Tensor Attribute: {m} (shape={attr.shape})")

print("="*50)

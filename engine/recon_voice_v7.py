import torch
import os
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_WAV = os.path.expanduser("~/.cache/axiom_vox/prime_voice/prime_reference.wav")

print("Loading model for v7...")
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

print(f"Conditionals status (pre): {hasattr(model, 'conds')} {type(getattr(model, 'conds', None))}")

print(f"Distilling from {REF_WAV}...")
model.prepare_conditionals(REF_WAV)

print(f"Conditionals status (post): {hasattr(model, 'conds')} {type(getattr(model, 'conds', None))}")

conds = getattr(model, "conds", None)
if conds is not None:
    if isinstance(conds, torch.Tensor):
        print(f"CONDS: shape={conds.shape}, dtype={conds.dtype}, sum={conds.sum()}")
    elif isinstance(conds, (list, tuple)):
        print(f"CONDS: length={len(conds)}, types={[type(x) for x in conds]}")
    elif isinstance(conds, dict):
        print(f"CONDS: keys={conds.keys()}")

# Test if we can SAVE it
SAVE_PATH = "/tmp/test_conds.pt"
try:
    torch.save(conds, SAVE_PATH)
    print(f"SUCCESS: Saved conds to {SAVE_PATH}")
except Exception as e:
    print(f"ERROR: Failed to save conds: {e}")

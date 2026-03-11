import sys
import os
import axiom_vox
print(f"PYTHON: {sys.executable}")
print(f"AXIOM_VOX: {axiom_vox.__file__}")
print(f"CWD: {os.getcwd()}")
print(f"PATH:\n" + "\n".join(sys.path))

# AXIØM VØX Integration

**Governed Text-to-Speech with Full AXIØM Intelligence**

VØX is a TTS system based on Qwen3-TTS. This integration layer wraps VØX with AXIØM's complete governance framework including:

- **Self-Model** - Identity boundaries and knowledge grounding
- **8 Universal Laws** - Unity, Polarity, Rhythm, Correspondence, Limitation, Emergence, Entropy, Propagation
- **Content Governance** - Grounding, tone, anti-tailing
- **Voice Boundaries** - Consent, identity protection, use-case restrictions
- **Prosody Guardrails** - Emotional expression governance
- **Persistent Storage** - SQLite for registries, Redis for rate limiting
- **Real Synthesis** - Qwen3-TTS integration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GOVERNED TTS API                              │
│  POST /synthesize                                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         VOX GOVERNOR                                 │
│                                                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │   SELF    │  │  AXIØM    │  │  CONTENT  │  │   VOICE   │       │
│  │   MODEL   │→ │   LAWS    │→ │GOVERNANCE │→ │ BOUNDARIES│       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
│        │              │              │              │               │
│        ▼              ▼              ▼              ▼               │
│   Identity       8 Laws         Grounding       Consent            │
│   Boundaries     Compliance     Tone            Identity           │
│   Knowledge      Coherence      Tailing         Use Case           │
│                                                                      │
│                        ┌───────────┐                                │
│                        │  PROSODY  │                                │
│                        │ GUARDRAILS│                                │
│                        └───────────┘                                │
│                              │                                      │
│                              ▼                                      │
│                         Emotion                                     │
│                         Manipulation                                │
│                         Brand Voice                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼ (if all pass)
┌─────────────────────────────────────────────────────────────────────┐
│                       VØX SYNTHESIZER                               │
│  Qwen3-TTS: Voice Clone / Voice Design / Streaming Synthesis        │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERSISTENT STORAGE                             │
│  SQLite: Voices, Consents, Audit Log  │  Redis: Rate Limiting      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## New in v0.2.0

### Self-Model Integration

Before speaking, VØX now asks itself:

```python
# Would speaking this violate my boundaries?
would_violate, boundary = self_model.would_violate(f"speak as {voice_id}: {text}")

# Can I actually do this?
can_speak, reason = self_model.can_do(f"speak: {voice_id}")

# Do I know this content is grounded?
knows, certainty = self_model.knows(content_claim)
```

### 8 AXIØM Laws Governance

Every speech request is checked against all 8 Universal Laws:

| Law | VØX Check |
|-----|-----------|
| **Unity** | Voice + Content + Emotion form coherent message |
| **Polarity** | Balanced warmth/authority, no sycophancy |
| **Rhythm** | Natural speech patterns, no run-on sentences |
| **Correspondence** | Micro-tone matches macro-tone |
| **Limitation** | No claims beyond scope |
| **Emergence** | Parts combine into coherent whole |
| **Entropy** | Voice drift detection |
| **Propagation** | Content returns value, builds trust |

### Persistent Storage

All registries now persist across restarts:

```python
from axiom_vox import get_database

db = get_database()  # SQLite at ~/.axiom_vox/vox.db

# Register voice with consent
db.register_voice("my_voice", category="consented", consent_verified=True)

# Grant consent with proof
db.grant_consent("my_voice", proof="signed_agreement.pdf")

# Audit log
db.log_audit(request_id="abc", request_type="synthesize", action="allow", passed=True)
```

### Real TTS Synthesis

Actual Qwen3-TTS integration:

```python
from axiom_vox import synthesize, VoiceConfig

# Quick synthesis
result = synthesize("Hello world", voice_id="axiom_default")
print(result.audio_data)  # Actual WAV bytes

# Custom voice config
config = VoiceConfig(
    voice_id="custom",
    speaking_rate=1.2,
    emotion="warm"
)
synth = VoxSynthesizer()
result = synth.synthesize("Hello", voice=config)
```

---

## Quick Start

### Basic Governance

```python
from axiom_vox import VoxGovernor, synthesize

# Create governor with all features
governor = VoxGovernor(
    strict_mode=False,
    require_self_model_check=True,
    require_laws_check=True,
    use_persistent_storage=True,
)

# Govern and synthesize
result = governor.govern(
    text="Welcome to our service.",
    voice_id="axiom_warm",
    context={"domain": "general"}
)

if result.action.value in ("allow", "repair"):
    audio = synthesize(result.governed_text, voice_id=result.voice_id)
    print(f"Audio generated: {len(audio.audio_data)} bytes")
else:
    print(f"Refused: {result.refusal_reason}")
```

### Check Law Violations

```python
from axiom_vox import check_axiom_laws

passes, violations = check_axiom_laws(
    text="BUY NOW! Trust me, this is 100% guaranteed!",
    voice_id="sales",
    context={"domain": "advertising"}
)

for v in violations:
    print(f"{v.law.value}: {v.description} (severity: {v.severity})")
    # limitation: Claim exceeds reasonable limits (severity: 0.6)
    # propagation: Content pattern undermines trust (severity: 0.6)
```

### API Server

```bash
# Start with governance
uvicorn axiom_vox.governed_tts_api:app --port 8000

# Synthesize
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "axiom_default"}'
```

---

## Components

| Module | Purpose |
|--------|---------|
| `vox_governor.py` | Main governance pipeline with Self-Model + Laws |
| `axiom_laws.py` | 8 Universal Laws compliance checks |
| `voice_boundaries.py` | Voice cloning ethics with persistent storage |
| `prosody_guardrails.py` | Emotional expression governance |
| `persistence.py` | SQLite database + Redis rate limiting |
| `synthesis.py` | Qwen3-TTS integration |
| `governed_tts_api.py` | FastAPI wrapper |

---

## Configuration

### Environment Variables

```bash
# Database
export AXIOM_VOX_DB_PATH="~/.axiom_vox/vox.db"

# Redis (optional, for rate limiting)
export REDIS_URL="redis://localhost:6379"

# API
export AXIOM_VOX_API_KEYS="key1,key2"
export AXIOM_VOX_STRICT_MODE=true

# TTS Model
export AXIOM_VOX_MODEL_SIZE="small"  # or "large" for 1.7B
```

### Programmatic

```python
governor = VoxGovernor(
    strict_mode=True,              # Refuse on any violation
    auto_repair=True,              # Auto-fix when possible
    require_self_model_check=True, # Use Self-Model
    require_laws_check=True,       # Check 8 Laws
    require_voice_clearance=True,  # Check voice boundaries
    require_prosody_approval=True, # Check emotional expression
    use_persistent_storage=True,   # Use SQLite
)
```

---

## Installation

```bash
# Core governance
pip install -e /path/to/axiom-kernel

# For API server
pip install fastapi uvicorn

# For TTS synthesis
pip install torch transformers soundfile numpy

# For Redis rate limiting (optional)
pip install redis
```

---

## Testing

```bash
# Run all demos
python -m axiom_vox.vox_governor
python -m axiom_vox.voice_boundaries
python -m axiom_vox.prosody_guardrails
python -m axiom_vox.axiom_laws
python -m axiom_vox.synthesis

# Start API
python -m axiom_vox.governed_tts_api
```

---

## Roadmap

- [x] Self-Model integration
- [x] 8 AXIØM Laws governance
- [x] SQLite persistent storage
- [x] Qwen3-TTS synthesis hook
- [ ] Voice biometric verification
- [ ] Real-time streaming prosody analysis
- [ ] Multi-language manipulation detection
- [ ] VØX fine-tuning pipeline integration
- [ ] Federated consent registry

---

## License

Part of the AXIØM Kernel. See repository root for license.

# AXIØM VØX

**Governed Voice Intelligence — Your Voice is the Interface**

## Structure

```
axiom-vox/
├── engine/     # Python TTS governance pipeline (Qwen3-TTS, 8 Laws, prosody guardrails)
├── site/       # Product landing page + Electron Mac app shell (Vite)
└── agents/     # Voice agents (VOICE_GENOME, VOICEPRINT, PRIME_VOICE)
```

## Engine

The governance layer that ensures every voice output passes AXIØM's 8 Universal Laws before synthesis. See [engine/README.md](engine/README.md).

## Site

The VØX product marketing site and Electron Mac app wrapper. See [site/](site/).

## Agents

Voice-specific agent configurations for voice genome analysis, voiceprint management, and PRIME voice identity.

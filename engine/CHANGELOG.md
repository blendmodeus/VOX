# Changelog

All notable changes to AXIØM VØX.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-03-15

AI Text Formatter + Production Hardening + Wake Word — Glaido Killer

### Added

- `TextFormatter` with 3 modes: raw / clean / professional
- Filler removal: 30+ patterns with context-aware "like" detection
- AI polish via gpt-4o-mini (professional mode)
- `format` param on `POST /transcribe` (backward compatible)
- FORMAT toggle in KITT dashboard (⚡raw / ✦clean / ◆pro)
- **Custom Dictionary** (`HotwordManager`) — domain terms + auto-corrections
- **Voice Snippets** (`SnippetManager`) — trigger phrases with dynamic vars
- **Wake Word** (`WakeWordDetector`) — "VOX" activation, no hotkey needed
- **Confidence Gate** — rejects garbage transcriptions below 40%
- **Auto-Paragraph** — inserts breaks on pauses >1.5s
- CRUD endpoints: `/stt/hotwords`, `/stt/snippets`, `/stt/wakeword/*`

### Fixed

- MODEL button now calls `POST /stt/load-model` (was JS-only)
- `/transcribe` now sends model, language, format params from KITT
- `electron-main.cjs` hardcoded paths → relative `path.resolve()`
- "like" filler detection: forward-looking context (keeps "I like pizza")
- Python path auto-detection with venv fallback

## [1.4.0] - 2026-03-11

Governed STT Engine — Gladia/Whisper Replacement

### Added

- `engine/stt/` module: governed speech-to-text pipeline
- `VoxTranscriber`: faster-whisper wrapper with model management, device auto-detection
- `STTGovernor`: post-transcription governance (PII redaction, content filtering, AXIØM Laws, confidence gating)
- `StreamingTranscriber`: real-time streaming transcription with session management
- `TranscriptionResult`: structured results with segments, word timestamps, confidence scores
- `POST /stt/load-model`: dynamic model switching from KITT overlay
- `GET /stt/models`: list available STT models
- `WebSocket /stt/stream`: streaming transcription endpoint
- Model catalog: tiny (75MB) → large-v3 (3GB)
- 35 new tests in `test_stt.py`

### Changed

- `POST /transcribe` now uses VoxTranscriber + STTGovernor (backward compatible)
- `vox_api.py` upgraded to v1.4.0 with governance pipeline

## [1.0.0] - 2026-01-31

Stable Release - All 12 AXIOM Phases Complete


### Added

- Complete AXIOM 12-Phase development cycle
- Production-ready TTS with governance
- Full SDK, verification, and documentation

### Changed

- Version bump to stable 1.0.0

## [0.17.0] - 2026-01-31

Release & Reflect Layer


### Added

- ReleaseManager for semantic versioning and changelog generation
- PackageBuilder for wheel/sdist validation and test install
- DeploymentHelper with Docker, Kubernetes, Lambda, Cloud Run configs
- SystemIntrospector for capability reporting and module analysis
- ReflectionEngine for usage analytics and improvement recommendations
- SemanticVersion parser with comparison and bumping

## [0.16.0] - 2026-01-31

Documentation & Examples Layer


### Added

- DocGenerator for automatic API documentation
- ExampleRunner with validation
- TutorialBuilder framework
- ChangelogManager with git integration
- MigrationGuide generator

## [0.15.0] - Unreleased

Verification Suite


### Added

- E2ETestRunner for end-to-end testing
- BenchmarkSuite for performance testing
- QualityValidator for audio validation
- HealthChecker with liveness/readiness probes

## [0.14.0] - Unreleased

Performance Layer


### Added

- AudioCache with LRU eviction
- EmbeddingCache with lazy loading
- HTTPConnectionPool and WebSocketPool
- BatchOptimizer with multiple strategies
- StreamBuffer with backpressure
- LazyLoader for heavy modules

## [0.13.0] - Unreleased

VØX Client SDK


### Added

- VoxClient high-level SDK
- RetryPolicy with exponential backoff
- VoxSession for request tracking
- Workflow helpers for common patterns

## [0.12.0] - Unreleased

Resource Governance Layer


### Added

- Sliding window rate limiting
- Token bucket rate limiting
- Tiered usage quotas
- Policy engine for content/usage policies
- SecurityManager with RBAC

## [0.11.0] - Unreleased

Unified Voice Pipeline


### Added

- VoxUnifiedPipeline single entry point
- BiometricVoiceRouter for intelligent routing
- RealTimeQualityMonitor
- UnifiedConsentRegistry

## [0.10.0] - Unreleased

Voice Biometric Verification


### Added

- VoiceBiometricService
- SpectralFingerprint 256-dim embeddings
- LivenessDetector for replay/deepfake detection
- DriftMonitor for voice changes

## [0.9.0] - Unreleased

Multi-voice Synthesis


### Added

- DialogueScript for multi-voice content
- CharacterRegistry for voice mapping
- TransitionProcessor for voice transitions
- MultiVoiceSynthesizer


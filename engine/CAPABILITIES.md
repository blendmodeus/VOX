# axiom_vox Capability Report

**Version:** 0.17.0
**Python:** 3.14.2
**Platform:** macOS-26.2-arm64-arm-64bit-Mach-O
**Generated:** 2026-01-31 23:14:35

## Components

- ✗ AnalyticsPipeline
- ✓ AudioCache
- ✓ BatchOptimizer
- ✓ BenchmarkSuite
- ✓ DialogueScript
- ✓ DocGenerator
- ✓ E2ETestRunner
- ✓ ExampleRunner
- ✗ GovernedTTS
- ✓ HealthChecker
- ✓ MultiVoiceSynthesizer
- ✓ PackageBuilder
- ✓ PolicyEngine
- ✓ QuotaManager
- ✓ RateLimiter
- ✓ ReleaseManager
- ✓ SpectralFingerprint
- ✗ StreamingAnalyzer
- ✗ StreamingPipeline
- ✓ TutorialRunner
- ✓ VoiceBiometricService
- ✗ VoiceCloneManager
- ✓ VoiceSpaceDirector
- ✓ VoxClient
- ✓ VoxSession
- ✓ VoxUnifiedPipeline

## Features

- ✓ api_documentation
- ✓ batch_optimization
- ✓ benchmarking
- ✓ biometric_verification
- ✓ caching
- ✓ connection_pooling
- ✓ e2e_testing
- ✓ health_checks
- ✓ liveness_detection
- ✓ multi_voice_dialogue
- ✓ policy_engine
- ✓ quota_management
- ✓ rate_limiting
- ✗ real_time_analytics
- ✓ runnable_examples
- ✗ streaming_synthesis
- ✓ tutorials
- ✗ voice_cloning
- ✓ voice_matching

## Dependencies

- aiohttp: not installed
- fastapi: not installed
- numpy: 2.4.1
- pydantic: 2.12.5
- scipy: not installed
- uvicorn: not installed

## Optional Dependencies

- librosa: ✗ not installed
- prometheus_client: ✗ not installed
- psutil: ✗ not installed
- soundfile: ✗ not installed
- speechbrain: ✗ not installed
- torch: ✗ not installed
- torchaudio: ✗ not installed
- transformers: ✗ not installed

## API Endpoints

- `GET /health`
- `GET /ready`
- `GET /voices`
- `POST /biometrics/enroll`
- `POST /biometrics/verify`
- `POST /stream`
- `POST /synthesize`
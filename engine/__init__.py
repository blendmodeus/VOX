"""
AXIØM VØX Integration
---------------------

Governed text-to-speech with AXIØM intelligence.

v1.3.0: PRIME Voice - Sovereign voice identity for PRIME agent (NEW)
v1.2.0: Voice Genome - Complete voice DNA mapping and analysis
v1.1.0: Resonance Analysis - Song psychological impact prediction
v1.0.0: Stable Release - All 12 AXIOM Phases Complete (STABLE)
v0.17.0: Release & Reflect Layer - Versioning, packaging, deployment, introspection
v0.16.0: Documentation Layer - Auto-docs, examples, tutorials, changelog
v0.15.0: Verification Suite - E2E testing, benchmarks, quality validation, health checks
v0.14.0: Performance Layer - Caching, connection pooling, batching, lazy loading
v0.13.0: VØX Client SDK - High-level async client with retry, sessions, workflows
v0.12.0: Resource Governance Layer - Rate limiting, quotas, policies, security
v0.11.0: Unified Voice Pipeline - Single entry point integrating all components
v0.10.0: Voice Biometric Verification - Identity authentication with liveness detection
v0.9.0: Multi-voice Synthesis - Seamless multi-voice dialogue with transitions

Components:
    - VoiceSpaceDirector: Multidimensional content→voice matching
    - ProsodyDirector: Law-derived prosody optimization
    - VoxGovernor: Governance pipeline (opt-in for compliance)
    - VoiceBoundaries: Ethical guardrails for voice cloning
    - VoxSynthesizer: Qwen3-TTS synthesis engine with adapter support
    - VoxDatabase: Persistent storage for registries
    - Fine-tuning: LoRA-based voice cloning pipeline
    - SSML: W3C SSML 1.1 parser and generator with <voice> tag support
    - Emotion Presets: Named emotion configurations
    - Analytics: Quality scoring, naturalness metrics, performance monitoring
    - Streaming Analytics: Per-chunk metrics, stutter detection, live monitoring
    - Multi-voice: DialogueScript, character registry, voice transitions
    - Biometrics: Voice enrollment, verification, liveness, drift detection
    - Unified: VoxUnifiedPipeline, BiometricVoiceRouter, RealTimeQualityMonitor
    - Governance: RateLimiter, QuotaManager, PolicyEngine, SecurityManager
    - SDK: VoxClient, VoxSession, RetryPolicy, Workflows
    - Performance: AudioCache, EmbeddingCache, ConnectionPool, BatchOptimizer, LazyLoader
    - Verification: E2ETestRunner, BenchmarkSuite, QualityValidator, HealthChecker
    - Documentation: DocGenerator, ExampleRunner, TutorialBuilder, ChangelogManager
    - Release: ReleaseManager, PackageBuilder, DeploymentHelper, SystemIntrospector, ReflectionEngine
    - Resonance: AudioAnalyzer, LyricAnalyzer, PsychoacousticMapper, ResonanceSynthesizer
    - Genome: VoiceGenomeSynthesizer, BiometricAnalyzer, PsychometricAnalyzer, SociometricAnalyzer
    - PRIME Voice: PrimeVoiceRenderer, PrimeVoiceStreamer, SpeakingModeManager, IdentityManager (NEW)

Features:
    - 8-dimensional voice space (formality, temperature, energy, authority, etc.)
    - Content analysis → voice vector → optimal voice match
    - Inflection parameters derived from dimensional correspondence
    - AXIØM Laws as generative principles, not just constraints
    - Voice cloning via LoRA adapters with consent verification
    - Job-based fine-tuning with progress tracking
    - SSML parsing: <break>, <emphasis>, <prosody>, <say-as>, <sub>, <voice>
    - 18 emotion presets: joy, sadness, calm, professional, warm, etc.
    - Streaming TTS with HTTP and WebSocket endpoints
    - Voice analytics: SNR, spectral analysis, naturalness estimation
    - Quality scoring with tiers (excellent/good/acceptable/poor)
    - Performance metrics: latency, RTF, throughput
    - Analytics API with trends and aggregation
    - Streaming analytics: per-chunk latency, stutter detection, quality drops
    - Live streaming metrics: real-time RTF, running quality score
    - Multi-voice synthesis: DialogueScript, SSML <voice> tags
    - Character-to-voice mapping with auto-assignment
    - Voice transitions: crossfade, breath pause, silence, immediate
    - Voice biometric enrollment and verification
    - 256-dim SpectralFingerprint speaker embeddings
    - Liveness detection: replay/deepfake prevention
    - Voice drift monitoring and adaptive templates
    - Unified voice pipeline: single entry point
    - Biometric voice routing: intelligent voice selection
    - Real-time quality monitoring during synthesis
    - Central consent registry across all operations
    - Sliding window and token bucket rate limiting (NEW)
    - Tiered usage quotas (free/basic/pro/enterprise) (NEW)
    - Content and usage policy enforcement (NEW)
    - Role-based access control and audit logging (NEW)
    - Resource allocation and queue management
    - VoxClient SDK for simplified API access (NEW)
    - Async-first design with sync wrappers (NEW)
    - Automatic retry with exponential backoff (NEW)
    - Session management with request correlation (NEW)
    - Workflow helpers for batch and dialogue synthesis
    - LRU audio caching with content-hash keys (NEW)
    - Embedding cache with lazy loading and prewarming (NEW)
    - HTTP/WebSocket connection pooling (NEW)
    - Intelligent batch processing with voice grouping (NEW)
    - Zero-copy streaming buffers with backpressure (NEW)
    - Lazy loading of heavy modules
    - E2E test framework with parallel execution (NEW)
    - Performance benchmarks with latency/throughput/memory (NEW)
    - Audio quality validation with SNR, silence, clipping detection (NEW)
    - System health monitoring with liveness/readiness probes
    - Auto-generate API documentation from docstrings (NEW)
    - Runnable code examples with validation (NEW)
    - Interactive step-by-step tutorials (NEW)
    - Changelog generation from git commits (NEW)
    - Migration guides for version upgrades
    - Semantic version management with auto-bumping (NEW)
    - Package building with wheel/sdist validation (NEW)
    - Deployment config generation: Docker, K8s, Lambda, Cloud Run (NEW)
    - System introspection with capability reporting (NEW)
    - Reflection engine with usage analytics and recommendations
    - Song audio feature extraction (tempo, key, spectrum, dynamics) (NEW)
    - Lyric analysis (sentiment, themes, repetition, imagery) (NEW)
    - Psychoacoustic effect mapping (features → psychological effects) (NEW)
    - Listener impact prediction with warnings
    - Voice DNA mapping: biometric, psychometric, sociometric analysis (NEW)
    - Age/sex estimation from acoustic features (NEW)
    - Vocal health assessment: fatigue, strain, hydration (NEW)
    - Emotional state detection with stress/anxiety markers (NEW)
    - Authenticity and deception analysis (NEW)
    - Big Five personality signal inference (NEW)
    - Social dynamics: authority, warmth, trust, charisma, persuasion
    - PRIME sovereign voice identity with locked 8-dim voice vector (NEW)
    - 7 context-aware speaking modes with auto-detection (NEW)
    - Real-time streaming voice with sentence-level chunking (NEW)
    - Biometric identity anchor for voice authenticity (NEW)
    - Mode transitions: briefing, alert, reflective, directive, empathetic, ceremonial (NEW)

Usage:
    from axiom_vox import VoiceSpaceDirector, synthesize

    # Multidimensional voice matching
    director = VoiceSpaceDirector()
    result = director.direct(
        text="We need to address this security issue immediately.",
        context={"domain": "technical"}
    )

    print(result["matched_voice_id"])  # "expert"
    print(result["inflections"])       # TTS parameters

    # Synthesize with matched voice and inflections
    audio = synthesize(
        text,
        voice_id=result["matched_voice_id"],
        **result["inflections"]
    )

    # Fine-tuning (voice cloning)
    from axiom_vox.finetuning import VoxFineTuningPipeline

    pipeline = VoxFineTuningPipeline(voice_id="my_voice")
    result = await pipeline.train(
        audio_samples=["voice1.wav", "voice2.wav"],
        consent_verified=True,
    )
    # Use cloned voice: voice_id="clone_xyz123"
"""

# Core governor
from axiom_vox.vox_governor import (
    VoxGovernor,
    VoxGovernanceResult,
    GovernanceAction,
    govern_speech,
    get_governor,
)

# Voice boundaries
from axiom_vox.voice_boundaries import (
    VoiceBoundaries,
    VoiceCloneRequest,
    CloneDecision,
    VoiceCategory,
    check_clone_ethics,
)

# Prosody guardrails (safety)
from axiom_vox.prosody_guardrails import (
    ProsodyGuardrails,
    EmotionalIntent,
    ProsodyDecision,
    EmotionCategory,
    govern_prosody,
)

# Prosody director (quality)
from axiom_vox.prosody_director import (
    ProsodyDirector,
    ProsodyTarget,
    ContentArchetype,
    PitchContour,
    direct_prosody,
    get_director,
)

# Voice space (multidimensional correspondence)
from axiom_vox.voice_space import (
    VoiceSpaceDirector,
    VoiceVector,
    VoiceProfile,
    direct_voice,
    get_voice_space_director,
)

# AXIØM Laws
from axiom_vox.axiom_laws import (
    AxiomLawsGovernor,
    AxiomLaw,
    LawViolation,
    check_axiom_laws,
)

# Persistence
from axiom_vox.persistence import (
    VoxDatabase,
    RateLimiter,
    get_database,
    get_rate_limiter,
)

# Synthesis
from axiom_vox.synthesis import (
    VoxSynthesizer,
    VoiceConfig,
    SynthesisResult,
    AudioFormat,
    synthesize,
    get_synthesizer,
)

# API
from axiom_vox.governed_tts_api import create_governed_tts_app

# SSML support
from axiom_vox.ssml import (
    SSMLParser,
    SSMLGenerator,
    SSMLDocument,
    SSMLBreak,
    SSMLEmphasis,
    SSMLProsody,
    SSMLSayAs,
    SSMLSub,
    SSMLVoice,
)

# Multi-voice synthesis (v0.9.0)
from axiom_vox.multi_voice import (
    DialogueLine,
    DialogueScript,
    VoiceSwitch,
    VoiceSwitchType,
    MultiVoiceSegment,
    MultiVoiceSynthesisResult,
    TransitionStyle,
    merge_consecutive_lines,
    parse_screenplay_format,
    parse_chat_format,
)

from axiom_vox.character_registry import (
    CharacterRegistry,
    CharacterVoiceMapping,
    get_character_registry,
    set_character_registry,
)

from axiom_vox.transition_processor import (
    TransitionProcessor,
    TransitionConfig,
    TransitionResult,
    TransitionStyle as TransitionStyleEnum,
    generate_breath_pause,
    apply_crossfade,
)

from axiom_vox.multi_voice_synthesizer import (
    MultiVoiceSynthesizer,
    MultiVoiceConfig,
    get_multi_voice_synthesizer,
    synthesize_dialogue,
)

# Emotion presets
from axiom_vox.emotion_presets import (
    EmotionPreset,
    EmotionPresetName,
    EMOTION_PRESETS,
    get_emotion_preset,
    create_intent_from_preset,
    list_emotion_presets,
    validate_preset_name,
)

# Analytics
from axiom_vox.analytics import (
    # Enums
    ComputationStatus,
    QualityTier,
    AnalyticsEventType,
    EventSeverity,
    # Quality metrics
    TechnicalQualityMetrics,
    SpectralQualityMetrics,
    NaturalnessMetrics,
    PerformanceMetrics,
    VoiceConsistencyMetrics,
    # Analytics containers
    SynthesisAnalytics,
    StreamingSessionAnalytics,
    VoiceAggregateMetrics,
    SystemAnalyticsSummary,
    AnalyticsEvent,
    # Components
    VoxAudioAnalyzer,
    VoxMetricsCollector,
    AnalyticsStorage,
    VoxAnalyticsService,
    # Streaming Analytics
    StreamingAnalyticsCollector,
    StreamChunkMetrics,
    StreamSessionMetrics,
    StreamingEvent,
    StreamingEventType,
    get_streaming_collector,
    set_streaming_collector,
    # Functions
    get_collector,
    set_collector,
    get_analytics_service,
    init_analytics_service,
    create_analytics_router,
)

# Biometrics (v0.10.0)
from axiom_vox.biometrics import (
    # Service
    VoiceBiometricService,
    get_biometric_service,
    enroll_voice,
    verify_voice,
    # Embeddings
    SpectralFingerprint,
    EmbeddingExtractor,
    get_extractor,
    # Liveness
    LivenessDetector,
    check_liveness,
    # Drift
    DriftMonitor,
    analyze_drift,
    # Models
    BiometricTemplate,
    BiometricConfig,
    EnrollmentResult,
    VerificationResult as BiometricVerificationResult,
    LivenessResult,
    DriftReport,
    EmbeddingBackend,
    EnrollmentStatus,
    VerificationStatus as BiometricVerificationStatus,
    LivenessStatus,
    DriftSeverity,
    # Consent
    BiometricConsentManager,
    ConsentType,
)

# Unified Pipeline (v0.11.0)
from axiom_vox.unified import (
    # Pipeline
    VoxUnifiedPipeline,
    get_unified_pipeline,
    synthesize_unified,
    # Models
    PipelineRequest,
    PipelineResponse,
    PipelineConfig,
    PipelineStage,
    PipelineStatus,
    # Consent Registry
    UnifiedConsentRegistry,
    get_consent_registry,
    ConsentScope,
    # Voice Router
    BiometricVoiceRouter,
    get_voice_router,
    VoiceRouteType,
    # Quality Monitor
    RealTimeQualityMonitor,
    get_quality_monitor,
    QualityGate,
)

# VØX Client SDK (v0.13.0)
from axiom_vox.sdk import (
    # Main client
    VoxClient,
    # Errors
    VoxError,
    ErrorCategory,
    ErrorContext,
    RetryStrategy,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    InvalidVoiceError,
    GovernanceError,
    ContentBlockedError,
    RateLimitError as SDKRateLimitError,
    QuotaExceededError as SDKQuotaExceededError,
    BiometricError,
    EnrollmentError,
    VerificationError as SDKVerificationError,
    LivenessError,
    NotEnrolledError,
    SynthesisError as SDKSynthesisError,
    QualityGateError,
    ResourceError,
    ResourceUnavailableError,
    NetworkError,
    TimeoutError as SDKTimeoutError,
    ConnectionError as SDKConnectionError,
    from_http_response,
    # Config
    VoxConfig,
    RetryConfig,
    TimeoutConfig,
    QualityConfig as SDKQualityConfig,
    GovernanceConfig as SDKGovernanceConfig,
    BiometricConfig as SDKBiometricConfig,
    Environment,
    LogLevel,
    DEFAULT_CONFIGS,
    get_default_config,
    # Retry
    RetryPolicy,
    RetryExecutor,
    RetryState,
    RateLimitHandler,
    with_retry,
    get_rate_limit_handler,
    # Session
    VoxSession,
    SessionPool,
    SessionState,
    SessionMetrics,
    RequestContext,
    # Workflows
    WorkflowStatus,
    WorkflowStep,
    WorkflowResult,
    SynthesisResult as SDKSynthesisResult,
    EnrollmentResult as SDKEnrollmentResult,
    VerificationResult as SDKVerificationResult,
    DialogueLine as SDKDialogueLine,
    DialogueResult,
    synthesize_with_quality_check,
    synthesize_verified,
    synthesize_batch,
    enroll_and_verify,
    continuous_verification,
    synthesize_dialogue as sdk_synthesize_dialogue,
    WorkflowBuilder,
    Workflow,
)

# Performance Layer (v0.14.0)
from axiom_vox.performance import (
    # Cache
    CacheStats,
    LRUCache,
    AudioCache,
    EmbeddingCache,
    CacheManager,
    get_cache_manager,
    set_cache_manager,
    # Pool
    ConnectionState,
    ConnectionStats as PoolConnectionStats,
    HTTPConnectionPool,
    WebSocketPool,
    ConnectionPoolManager,
    get_pool_manager,
    set_pool_manager,
    create_pool_manager,
    # Batch
    BatchStrategy,
    BatchStats,
    BatchItem,
    BatchResult,
    BatchOptimizer,
    SynthesisBatchOptimizer,
    RequestBatcher,
    get_batch_optimizer,
    set_batch_optimizer,
    # Stream
    BufferState,
    BufferStats,
    Chunk,
    StreamBuffer,
    AudioStreamBuffer,
    StreamPipeline,
    get_audio_buffer,
    set_audio_buffer,
    # Lazy
    LoadState,
    LoadStats,
    LazyLoader,
    ModuleLazyLoader,
    LazyAttribute,
    lazy_import,
    get_resource_loader,
    get_module_loader,
)

# Verification Suite (v0.15.0)
from axiom_vox.verification import (
    # Enums
    TestStatus,
    TestSeverity,
    BenchmarkType,
    HealthStatus,
    # Test models
    TestCase,
    TestResult,
    TestSuiteResult,
    # Benchmark models
    BenchmarkMetric,
    BenchmarkResult,
    # Quality models
    QualityMetric,
    QualityResult,
    # Health models
    HealthCheckResult,
    SystemHealthReport,
    # Report
    VerificationReport,
    # E2E testing
    E2ETestConfig,
    E2ETestRunner,
    create_vox_e2e_tests,
    run_quick_verification,
    run_full_verification,
    # Benchmarking
    BenchmarkConfig,
    BenchmarkSuite,
    create_vox_benchmarks,
    run_benchmarks,
    run_latency_benchmarks,
    # Quality
    QualityThresholds,
    QualityValidator,
    IntelligibilityEstimator,
    create_quality_validator,
    # Health
    HealthCheckConfig,
    HealthChecker,
    LivenessProbe,
    ReadinessProbe,
    create_vox_health_checker,
    run_health_check,
    check_system_ready,
)

# Documentation Layer (v0.16.0)
from axiom_vox.docs import (
    # Enums
    DocType,
    DocFormat,
    ExampleStatus,
    ChangeType,
    TutorialLevel,
    # Documentation models
    Parameter as DocParameter,
    DocEntry,
    ModuleDoc,
    # Example models
    Example,
    ExampleResult,
    # Tutorial models
    Tutorial,
    TutorialStep,
    TutorialProgress,
    # Changelog models
    ChangelogEntry,
    VersionChangelog,
    Changelog as DocChangelog,
    MigrationStep,
    MigrationGuide,
    # Generator
    GeneratorConfig,
    DocGenerator,
    generate_api_docs,
    # Examples
    ExampleConfig,
    ExampleRunner,
    create_vox_examples,
    run_examples,
    # Tutorials
    TutorialBuilder,
    TutorialRunner,
    create_vox_tutorials,
    # Changelog
    ChangelogConfig,
    ChangelogManager,
    create_vox_changelog,
)

# Release Layer (v0.17.0)
from axiom_vox.release import (
    # Enums
    ReleaseType,
    ReleaseStatus,
    DeploymentTarget,
    PackageFormat,
    ReflectionCategory,
    # Version
    SemanticVersion,
    # Release models
    ReleaseInfo,
    # Package models
    PackageMetadata,
    # Deployment models
    DeploymentConfig as ReleaseDeploymentConfig,
    # Introspection models
    CapabilityReport,
    # Reflection models
    ReflectionInsight,
    ReflectionReport,
    # Manager
    ReleaseConfig,
    ReleaseManager,
    create_release,
    get_current_version,
    # Packaging
    PackageConfig,
    ValidationResult,
    PackageArtifact,
    PackageBuilder,
    build_package,
    validate_package,
    # Deploy
    DeploymentResult,
    DeploymentHelper,
    generate_deployment_configs,
    deploy_docker,
    # Introspect
    IntrospectionConfig,
    ModuleInfo,
    DependencyInfo,
    SystemIntrospector,
    generate_capability_report,
    analyze_vox_structure,
    print_capability_report,
    # Reflect
    ReflectionConfig,
    UsageMetrics,
    PerformanceMetrics,
    ErrorMetrics,
    ReflectionEngine,
    generate_reflection_report,
    print_reflection_report,
)

# Resonance Analysis (v1.1.0)
from axiom_vox.resonance import (
    # Enums
    MusicalKey,
    EmotionalValence,
    ArousalLevel,
    ResonanceCategory,
    RiskLevel,
    # Feature models
    AudioFeatures,
    LyricFeatures,
    # Effect models
    PsychologicalEffect,
    ResonanceWarning,
    # Main profile
    ResonanceProfile,
    # Audio analyzer
    AudioAnalyzerConfig,
    AudioAnalyzer,
    analyze_audio,
    # Lyric analyzer
    LyricAnalyzerConfig,
    LyricAnalyzer,
    analyze_lyrics,
    # Psychoacoustic mapper
    PsychoacousticConfig,
    PsychoacousticMapper,
    map_psychoacoustic_effects,
    # Synthesizer
    ResonanceSynthesizerConfig,
    ResonanceSynthesizer,
    analyze_song,
    analyze_lyrics_only,
    generate_resonance_report,
)

# Voice Genome (v1.2.0)
from axiom_vox.genome import (
    # Enums
    HealthRisk,
    ConfidenceLevel,
    EmotionalState,
    AuthenticityLevel,
    DominanceLevel,
    # Feature models
    AcousticFeatures,
    # Marker models
    BiometricMarkers,
    PsychometricMarkers,
    SociometricMarkers,
    # Main genome
    VoiceGenome,
    GenomeComparison,
    # Extractor
    ExtractorConfig,
    VoiceFeatureExtractor,
    extract_voice_features,
    # Biometric
    BiometricConfig as GenomeBiometricConfig,
    BiometricAnalyzer,
    analyze_biometrics,
    # Psychometric
    PsychometricConfig,
    PsychometricAnalyzer,
    analyze_psychometrics,
    # Sociometric
    SociometricConfig,
    SociometricAnalyzer,
    analyze_sociometrics,
    # Synthesizer
    GenomeSynthesizerConfig,
    VoiceGenomeSynthesizer,
    analyze_voice_genome,
    generate_genome_report,
)

# PRIME Voice (v1.3.0)
from axiom_vox.prime_voice import (
    # Enums
    SpeakingModeType,
    PrimeVoiceState,
    IdentityLockLevel,
    UtteranceType,
    StreamEventType,
    # Identity models
    PrimeVocalDNA,
    PrimeVoiceVector,
    PrimeVoiceIdentity,
    # Mode models
    SpeakingModeProfile,
    ModeTransition,
    # Utterance models
    PrimeUtterance,
    PrimeVoiceSession,
    PrimeVoiceConfig,
    # Identity manager
    IdentityConfig,
    IdentityCheckResult,
    PrimeVoiceIdentityManager,
    get_identity_manager,
    get_prime_identity,
    get_prime_synthesis_params,
    # Speaking modes
    ModeDetectionResult,
    SpeakingModeManager,
    PRIME_SPEAKING_MODES,
    detect_speaking_mode,
    get_mode_profile as get_speaking_mode_profile,
    # Renderer
    RendererConfig,
    RenderResult,
    PrimeVoiceRenderer,
    get_renderer as get_prime_renderer,
    prime_speak,
    # Streamer
    PrimeAudioChunk,
    PrimeStreamEvent,
    StreamConfig as PrimeStreamConfig,
    PrimeVoiceStreamer,
    get_streamer as get_prime_streamer,
    prime_stream,
    segment_sentences,
)

# Resource Governance (v0.12.0)
from axiom_vox.governance import (
    # Enums
    PolicyType,
    PolicyAction,
    ResourceType,
    QuotaPeriod,
    SecurityLevel,
    ViolationType,
    # Rate Limiting
    RateLimitConfig,
    RateLimitResult,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    get_rate_limiter,
    # Resources
    ResourceConfig,
    ResourceStatus,
    ResourceManager,
    QueueManager,
    get_resource_manager,
    # Policies
    Policy,
    PolicyResult,
    ContentPolicy,
    UsagePolicy,
    PolicyEngine,
    get_policy_engine,
    # Quotas
    QuotaConfig,
    QuotaStatus,
    QuotaManager,
    get_quota_manager,
    # Security
    SecurityConfig,
    AccessToken,
    AuditEntry,
    SecurityManager,
    AccessController,
    AuditLogger,
    get_security_manager,
    # Defaults
    RATE_LIMIT_DEFAULTS,
    RESOURCE_LIMITS,
    QUOTA_DEFAULTS,
)

# Fine-tuning (lazy imports to avoid heavy dependencies)
def _import_finetuning():
    """Import fine-tuning module on demand."""
    from axiom_vox.finetuning import (
        VoxFineTuningPipeline,
        FineTuningConfig,
        FineTuningResult,
        VoxLoRAAdapter,
        LoRAConfig,
        VoxCheckpointManager,
        AudioProcessor,
        AudioSample,
        VoiceVerifier,
        VerificationResult,
        FineTuningJobManager,
        JobStatus,
        JobInfo,
    )
    return {
        "VoxFineTuningPipeline": VoxFineTuningPipeline,
        "FineTuningConfig": FineTuningConfig,
        "FineTuningResult": FineTuningResult,
        "VoxLoRAAdapter": VoxLoRAAdapter,
        "LoRAConfig": LoRAConfig,
        "VoxCheckpointManager": VoxCheckpointManager,
        "AudioProcessor": AudioProcessor,
        "AudioSample": AudioSample,
        "VoiceVerifier": VoiceVerifier,
        "VerificationResult": VerificationResult,
        "FineTuningJobManager": FineTuningJobManager,
        "JobStatus": JobStatus,
        "JobInfo": JobInfo,
    }


def __getattr__(name):
    """Lazy loading for fine-tuning module."""
    finetuning_exports = {
        "VoxFineTuningPipeline",
        "FineTuningConfig",
        "FineTuningResult",
        "VoxLoRAAdapter",
        "LoRAConfig",
        "VoxCheckpointManager",
        "AudioProcessor",
        "AudioSample",
        "VoiceVerifier",
        "VerificationResult",
        "FineTuningJobManager",
        "JobStatus",
        "JobInfo",
    }
    if name in finetuning_exports:
        return _import_finetuning()[name]
    raise AttributeError(f"module 'axiom_vox' has no attribute '{name}'")


__all__ = [
    # Core governor
    "VoxGovernor",
    "VoxGovernanceResult",
    "GovernanceAction",
    "govern_speech",
    "get_governor",
    # Voice boundaries
    "VoiceBoundaries",
    "VoiceCloneRequest",
    "CloneDecision",
    "VoiceCategory",
    "check_clone_ethics",
    # Prosody guardrails (safety)
    "ProsodyGuardrails",
    "EmotionalIntent",
    "ProsodyDecision",
    "EmotionCategory",
    "govern_prosody",
    # Prosody director (quality)
    "ProsodyDirector",
    "ProsodyTarget",
    "ContentArchetype",
    "PitchContour",
    "direct_prosody",
    "get_director",
    # Voice space (multidimensional)
    "VoiceSpaceDirector",
    "VoiceVector",
    "VoiceProfile",
    "direct_voice",
    "get_voice_space_director",
    # Laws
    "AxiomLawsGovernor",
    "AxiomLaw",
    "LawViolation",
    "check_axiom_laws",
    # Persistence
    "VoxDatabase",
    "RateLimiter",
    "get_database",
    "get_rate_limiter",
    # Synthesis
    "VoxSynthesizer",
    "VoiceConfig",
    "SynthesisResult",
    "AudioFormat",
    "synthesize",
    "get_synthesizer",
    # API
    "create_governed_tts_app",
    # SSML
    "SSMLParser",
    "SSMLGenerator",
    "SSMLDocument",
    "SSMLBreak",
    "SSMLEmphasis",
    "SSMLProsody",
    "SSMLSayAs",
    "SSMLSub",
    "SSMLVoice",
    # Multi-voice (v0.9.0)
    "DialogueLine",
    "DialogueScript",
    "VoiceSwitch",
    "VoiceSwitchType",
    "MultiVoiceSegment",
    "MultiVoiceSynthesisResult",
    "TransitionStyle",
    "merge_consecutive_lines",
    "parse_screenplay_format",
    "parse_chat_format",
    "CharacterRegistry",
    "CharacterVoiceMapping",
    "get_character_registry",
    "set_character_registry",
    "TransitionProcessor",
    "TransitionConfig",
    "TransitionResult",
    "generate_breath_pause",
    "apply_crossfade",
    "MultiVoiceSynthesizer",
    "MultiVoiceConfig",
    "get_multi_voice_synthesizer",
    "synthesize_dialogue",
    # Emotion presets
    "EmotionPreset",
    "EmotionPresetName",
    "EMOTION_PRESETS",
    "get_emotion_preset",
    "create_intent_from_preset",
    "list_emotion_presets",
    "validate_preset_name",
    # Analytics
    "ComputationStatus",
    "QualityTier",
    "AnalyticsEventType",
    "EventSeverity",
    "TechnicalQualityMetrics",
    "SpectralQualityMetrics",
    "NaturalnessMetrics",
    "PerformanceMetrics",
    "VoiceConsistencyMetrics",
    "SynthesisAnalytics",
    "StreamingSessionAnalytics",
    "VoiceAggregateMetrics",
    "SystemAnalyticsSummary",
    "AnalyticsEvent",
    "VoxAudioAnalyzer",
    "VoxMetricsCollector",
    "AnalyticsStorage",
    "VoxAnalyticsService",
    "get_collector",
    "set_collector",
    "get_analytics_service",
    "init_analytics_service",
    "create_analytics_router",
    # Streaming Analytics
    "StreamingAnalyticsCollector",
    "StreamChunkMetrics",
    "StreamSessionMetrics",
    "StreamingEvent",
    "StreamingEventType",
    "get_streaming_collector",
    "set_streaming_collector",
    # Fine-tuning (lazy loaded)
    "VoxFineTuningPipeline",
    "FineTuningConfig",
    "FineTuningResult",
    "VoxLoRAAdapter",
    "LoRAConfig",
    "VoxCheckpointManager",
    "AudioProcessor",
    "AudioSample",
    "VoiceVerifier",
    "VerificationResult",
    "FineTuningJobManager",
    "JobStatus",
    "JobInfo",
    # Biometrics (v0.10.0)
    "VoiceBiometricService",
    "get_biometric_service",
    "enroll_voice",
    "verify_voice",
    "SpectralFingerprint",
    "EmbeddingExtractor",
    "get_extractor",
    "LivenessDetector",
    "check_liveness",
    "DriftMonitor",
    "analyze_drift",
    "BiometricTemplate",
    "BiometricConfig",
    "EnrollmentResult",
    "BiometricVerificationResult",
    "LivenessResult",
    "DriftReport",
    "EmbeddingBackend",
    "EnrollmentStatus",
    "BiometricVerificationStatus",
    "LivenessStatus",
    "DriftSeverity",
    "BiometricConsentManager",
    "ConsentType",
    # Unified Pipeline (v0.11.0)
    "VoxUnifiedPipeline",
    "get_unified_pipeline",
    "synthesize_unified",
    "PipelineRequest",
    "PipelineResponse",
    "PipelineConfig",
    "PipelineStage",
    "PipelineStatus",
    "UnifiedConsentRegistry",
    "get_consent_registry",
    "ConsentScope",
    "BiometricVoiceRouter",
    "get_voice_router",
    "VoiceRouteType",
    "RealTimeQualityMonitor",
    "get_quality_monitor",
    "QualityGate",
    # Resource Governance (v0.12.0)
    "PolicyType",
    "PolicyAction",
    "ResourceType",
    "QuotaPeriod",
    "SecurityLevel",
    "ViolationType",
    "RateLimitConfig",
    "RateLimitResult",
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "get_rate_limiter",
    "ResourceConfig",
    "ResourceStatus",
    "ResourceManager",
    "QueueManager",
    "get_resource_manager",
    "Policy",
    "PolicyResult",
    "ContentPolicy",
    "UsagePolicy",
    "PolicyEngine",
    "get_policy_engine",
    "QuotaConfig",
    "QuotaStatus",
    "QuotaManager",
    "get_quota_manager",
    "SecurityConfig",
    "AccessToken",
    "AuditEntry",
    "SecurityManager",
    "AccessController",
    "AuditLogger",
    "get_security_manager",
    "RATE_LIMIT_DEFAULTS",
    "RESOURCE_LIMITS",
    "QUOTA_DEFAULTS",
    # VØX Client SDK (v0.13.0)
    "VoxClient",
    "VoxError",
    "ErrorCategory",
    "ErrorContext",
    "RetryStrategy",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "InvalidVoiceError",
    "GovernanceError",
    "ContentBlockedError",
    "SDKRateLimitError",
    "SDKQuotaExceededError",
    "BiometricError",
    "EnrollmentError",
    "SDKVerificationError",
    "LivenessError",
    "NotEnrolledError",
    "SDKSynthesisError",
    "QualityGateError",
    "ResourceError",
    "ResourceUnavailableError",
    "NetworkError",
    "SDKTimeoutError",
    "SDKConnectionError",
    "from_http_response",
    "VoxConfig",
    "RetryConfig",
    "TimeoutConfig",
    "SDKQualityConfig",
    "SDKGovernanceConfig",
    "SDKBiometricConfig",
    "Environment",
    "LogLevel",
    "DEFAULT_CONFIGS",
    "get_default_config",
    "RetryPolicy",
    "RetryExecutor",
    "RetryState",
    "RateLimitHandler",
    "with_retry",
    "get_rate_limit_handler",
    "VoxSession",
    "SessionPool",
    "SessionState",
    "SessionMetrics",
    "RequestContext",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowResult",
    "SDKSynthesisResult",
    "SDKEnrollmentResult",
    "SDKVerificationResult",
    "SDKDialogueLine",
    "DialogueResult",
    "synthesize_with_quality_check",
    "synthesize_verified",
    "synthesize_batch",
    "enroll_and_verify",
    "continuous_verification",
    "sdk_synthesize_dialogue",
    "WorkflowBuilder",
    "Workflow",
    # Performance Layer (v0.14.0)
    "CacheStats",
    "LRUCache",
    "AudioCache",
    "EmbeddingCache",
    "CacheManager",
    "get_cache_manager",
    "set_cache_manager",
    "ConnectionState",
    "PoolConnectionStats",
    "HTTPConnectionPool",
    "WebSocketPool",
    "ConnectionPoolManager",
    "get_pool_manager",
    "set_pool_manager",
    "create_pool_manager",
    "BatchStrategy",
    "BatchStats",
    "BatchItem",
    "BatchResult",
    "BatchOptimizer",
    "SynthesisBatchOptimizer",
    "RequestBatcher",
    "get_batch_optimizer",
    "set_batch_optimizer",
    "BufferState",
    "BufferStats",
    "Chunk",
    "StreamBuffer",
    "AudioStreamBuffer",
    "StreamPipeline",
    "get_audio_buffer",
    "set_audio_buffer",
    "LoadState",
    "LoadStats",
    "LazyLoader",
    "ModuleLazyLoader",
    "LazyAttribute",
    "lazy_import",
    "get_resource_loader",
    "get_module_loader",
    # Verification Suite (v0.15.0)
    "TestStatus",
    "TestSeverity",
    "BenchmarkType",
    "HealthStatus",
    "TestCase",
    "TestResult",
    "TestSuiteResult",
    "BenchmarkMetric",
    "BenchmarkResult",
    "QualityMetric",
    "QualityResult",
    "HealthCheckResult",
    "SystemHealthReport",
    "VerificationReport",
    "E2ETestConfig",
    "E2ETestRunner",
    "create_vox_e2e_tests",
    "run_quick_verification",
    "run_full_verification",
    "BenchmarkConfig",
    "BenchmarkSuite",
    "create_vox_benchmarks",
    "run_benchmarks",
    "run_latency_benchmarks",
    "QualityThresholds",
    "QualityValidator",
    "IntelligibilityEstimator",
    "create_quality_validator",
    "HealthCheckConfig",
    "HealthChecker",
    "LivenessProbe",
    "ReadinessProbe",
    "create_vox_health_checker",
    "run_health_check",
    "check_system_ready",
    # Documentation Layer (v0.16.0)
    "DocType",
    "DocFormat",
    "ExampleStatus",
    "ChangeType",
    "TutorialLevel",
    "DocParameter",
    "DocEntry",
    "ModuleDoc",
    "Example",
    "ExampleResult",
    "Tutorial",
    "TutorialStep",
    "TutorialProgress",
    "ChangelogEntry",
    "VersionChangelog",
    "DocChangelog",
    "MigrationStep",
    "MigrationGuide",
    "GeneratorConfig",
    "DocGenerator",
    "generate_api_docs",
    "ExampleConfig",
    "ExampleRunner",
    "create_vox_examples",
    "run_examples",
    "TutorialBuilder",
    "TutorialRunner",
    "create_vox_tutorials",
    "ChangelogConfig",
    "ChangelogManager",
    "create_vox_changelog",
    # Release Layer (v0.17.0)
    "ReleaseType",
    "ReleaseStatus",
    "DeploymentTarget",
    "PackageFormat",
    "ReflectionCategory",
    "SemanticVersion",
    "ReleaseInfo",
    "PackageMetadata",
    "ReleaseDeploymentConfig",
    "CapabilityReport",
    "ReflectionInsight",
    "ReflectionReport",
    "ReleaseConfig",
    "ReleaseManager",
    "create_release",
    "get_current_version",
    "PackageConfig",
    "ValidationResult",
    "PackageArtifact",
    "PackageBuilder",
    "build_package",
    "validate_package",
    "DeploymentResult",
    "DeploymentHelper",
    "generate_deployment_configs",
    "deploy_docker",
    "IntrospectionConfig",
    "ModuleInfo",
    "DependencyInfo",
    "SystemIntrospector",
    "generate_capability_report",
    "analyze_vox_structure",
    "print_capability_report",
    "ReflectionConfig",
    "UsageMetrics",
    "PerformanceMetrics",
    "ErrorMetrics",
    "ReflectionEngine",
    "generate_reflection_report",
    "print_reflection_report",
    # Resonance Analysis (v1.1.0)
    "MusicalKey",
    "EmotionalValence",
    "ArousalLevel",
    "ResonanceCategory",
    "RiskLevel",
    "AudioFeatures",
    "LyricFeatures",
    "PsychologicalEffect",
    "ResonanceWarning",
    "ResonanceProfile",
    "AudioAnalyzerConfig",
    "AudioAnalyzer",
    "analyze_audio",
    "LyricAnalyzerConfig",
    "LyricAnalyzer",
    "analyze_lyrics",
    "PsychoacousticConfig",
    "PsychoacousticMapper",
    "map_psychoacoustic_effects",
    "ResonanceSynthesizerConfig",
    "ResonanceSynthesizer",
    "analyze_song",
    "analyze_lyrics_only",
    "generate_resonance_report",
    # Voice Genome (v1.2.0)
    "HealthRisk",
    "ConfidenceLevel",
    "EmotionalState",
    "AuthenticityLevel",
    "DominanceLevel",
    "AcousticFeatures",
    "BiometricMarkers",
    "PsychometricMarkers",
    "SociometricMarkers",
    "VoiceGenome",
    "GenomeComparison",
    "ExtractorConfig",
    "VoiceFeatureExtractor",
    "extract_voice_features",
    "GenomeBiometricConfig",
    "BiometricAnalyzer",
    "analyze_biometrics",
    "PsychometricConfig",
    "PsychometricAnalyzer",
    "analyze_psychometrics",
    "SociometricConfig",
    "SociometricAnalyzer",
    "analyze_sociometrics",
    "GenomeSynthesizerConfig",
    "VoiceGenomeSynthesizer",
    "analyze_voice_genome",
    "generate_genome_report",
    # PRIME Voice (v1.3.0)
    "SpeakingModeType",
    "PrimeVoiceState",
    "IdentityLockLevel",
    "UtteranceType",
    "StreamEventType",
    "PrimeVocalDNA",
    "PrimeVoiceVector",
    "PrimeVoiceIdentity",
    "SpeakingModeProfile",
    "ModeTransition",
    "PrimeUtterance",
    "PrimeVoiceSession",
    "PrimeVoiceConfig",
    "IdentityConfig",
    "IdentityCheckResult",
    "PrimeVoiceIdentityManager",
    "get_identity_manager",
    "get_prime_identity",
    "get_prime_synthesis_params",
    "ModeDetectionResult",
    "SpeakingModeManager",
    "PRIME_SPEAKING_MODES",
    "detect_speaking_mode",
    "get_speaking_mode_profile",
    "RendererConfig",
    "RenderResult",
    "PrimeVoiceRenderer",
    "get_prime_renderer",
    "prime_speak",
    "PrimeAudioChunk",
    "PrimeStreamEvent",
    "PrimeStreamConfig",
    "PrimeVoiceStreamer",
    "get_prime_streamer",
    "prime_stream",
    "segment_sentences",
]

__version__ = "1.3.0"

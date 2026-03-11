"""
VØX Documentation - Tutorials
-----------------------------

Interactive tutorial framework.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable

from .models import (
    Tutorial,
    TutorialStep,
    TutorialLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class TutorialProgress:
    """
    Progress through a tutorial.

    Attributes:
        tutorial: The tutorial being followed
        current_step: Current step index
        completed_steps: List of completed step indices
        checkpoints_passed: Passed checkpoint names
        started_at: Start timestamp
    """
    tutorial: Tutorial
    current_step: int = 0
    completed_steps: List[int] = field(default_factory=list)
    checkpoints_passed: List[str] = field(default_factory=list)
    started_at: float = 0.0
    notes: Dict[int, str] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if tutorial is complete."""
        return len(self.completed_steps) == len(self.tutorial.steps)

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if not self.tutorial.steps:
            return 100.0
        return len(self.completed_steps) / len(self.tutorial.steps) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tutorial": self.tutorial.title,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "progress_percent": self.progress_percent,
            "is_complete": self.is_complete,
        }


class TutorialBuilder:
    """
    Builder for creating tutorials.

    Features:
        - Step-by-step construction
        - Checkpoint validation
        - Multiple difficulty levels
        - Export to multiple formats
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        level: TutorialLevel = TutorialLevel.BEGINNER,
    ):
        """
        Initialize tutorial builder.

        Args:
            title: Tutorial title
            description: Tutorial description
            level: Difficulty level
        """
        self._tutorial = Tutorial(
            title=title,
            description=description,
            level=level,
        )

    def set_level(self, level: TutorialLevel) -> "TutorialBuilder":
        """Set difficulty level."""
        self._tutorial.level = level
        return self

    def set_time(self, minutes: int) -> "TutorialBuilder":
        """Set estimated time."""
        self._tutorial.estimated_time_minutes = minutes
        return self

    def add_prerequisite(self, prereq: str) -> "TutorialBuilder":
        """Add a prerequisite."""
        self._tutorial.prerequisites.append(prereq)
        return self

    def add_tag(self, tag: str) -> "TutorialBuilder":
        """Add a tag."""
        self._tutorial.tags.append(tag)
        return self

    def add_step(
        self,
        title: str,
        content: str,
        code: str = "",
        expected_result: str = "",
        checkpoint: Optional[str] = None,
        hints: Optional[List[str]] = None,
    ) -> "TutorialBuilder":
        """
        Add a tutorial step.

        Args:
            title: Step title
            content: Explanation content
            code: Code to demonstrate
            expected_result: What should happen
            checkpoint: Validation checkpoint
            hints: Optional hints

        Returns:
            Self for chaining
        """
        step = TutorialStep(
            title=title,
            content=content,
            code=code,
            expected_result=expected_result,
            checkpoint=checkpoint,
            hints=hints or [],
        )
        self._tutorial.steps.append(step)
        return self

    def build(self) -> Tutorial:
        """
        Build the tutorial.

        Returns:
            Complete tutorial
        """
        return self._tutorial


class TutorialRunner:
    """
    Interactive tutorial runner.

    Features:
        - Step-by-step execution
        - Progress tracking
        - Checkpoint validation
        - Hint system
    """

    def __init__(self):
        """Initialize tutorial runner."""
        self._tutorials: Dict[str, Tutorial] = {}
        self._progress: Dict[str, TutorialProgress] = {}

    def register(self, tutorial: Tutorial) -> None:
        """
        Register a tutorial.

        Args:
            tutorial: Tutorial to register
        """
        self._tutorials[tutorial.title] = tutorial

    def list_tutorials(
        self,
        level: Optional[TutorialLevel] = None,
        tag: Optional[str] = None,
    ) -> List[Tutorial]:
        """
        List available tutorials.

        Args:
            level: Filter by level
            tag: Filter by tag

        Returns:
            Matching tutorials
        """
        tutorials = list(self._tutorials.values())

        if level:
            tutorials = [t for t in tutorials if t.level == level]

        if tag:
            tutorials = [t for t in tutorials if tag in t.tags]

        return tutorials

    def start(self, tutorial_title: str) -> TutorialProgress:
        """
        Start a tutorial.

        Args:
            tutorial_title: Tutorial to start

        Returns:
            Progress tracker
        """
        tutorial = self._tutorials.get(tutorial_title)
        if not tutorial:
            raise ValueError(f"Tutorial not found: {tutorial_title}")

        import time
        progress = TutorialProgress(
            tutorial=tutorial,
            started_at=time.time(),
        )
        self._progress[tutorial_title] = progress
        return progress

    def get_current_step(self, tutorial_title: str) -> Optional[TutorialStep]:
        """
        Get current step for a tutorial.

        Args:
            tutorial_title: Tutorial name

        Returns:
            Current step or None
        """
        progress = self._progress.get(tutorial_title)
        if not progress:
            return None

        if progress.current_step >= len(progress.tutorial.steps):
            return None

        return progress.tutorial.steps[progress.current_step]

    def complete_step(
        self,
        tutorial_title: str,
        validate: bool = True,
    ) -> bool:
        """
        Mark current step as complete.

        Args:
            tutorial_title: Tutorial name
            validate: Validate checkpoint if any

        Returns:
            True if step completed
        """
        progress = self._progress.get(tutorial_title)
        if not progress:
            return False

        step_idx = progress.current_step
        if step_idx >= len(progress.tutorial.steps):
            return False

        step = progress.tutorial.steps[step_idx]

        # Validate checkpoint if required
        if validate and step.checkpoint:
            if step.checkpoint not in progress.checkpoints_passed:
                logger.warning(f"Checkpoint not passed: {step.checkpoint}")
                return False

        # Mark complete
        if step_idx not in progress.completed_steps:
            progress.completed_steps.append(step_idx)

        # Advance to next step
        progress.current_step = step_idx + 1

        return True

    def pass_checkpoint(
        self,
        tutorial_title: str,
        checkpoint: str,
    ) -> bool:
        """
        Mark a checkpoint as passed.

        Args:
            tutorial_title: Tutorial name
            checkpoint: Checkpoint name

        Returns:
            True if checkpoint recorded
        """
        progress = self._progress.get(tutorial_title)
        if not progress:
            return False

        if checkpoint not in progress.checkpoints_passed:
            progress.checkpoints_passed.append(checkpoint)

        return True

    def get_hint(
        self,
        tutorial_title: str,
        hint_index: int = 0,
    ) -> Optional[str]:
        """
        Get a hint for current step.

        Args:
            tutorial_title: Tutorial name
            hint_index: Which hint to get

        Returns:
            Hint text or None
        """
        step = self.get_current_step(tutorial_title)
        if not step or not step.hints:
            return None

        if hint_index >= len(step.hints):
            return None

        return step.hints[hint_index]

    def get_progress(self, tutorial_title: str) -> Optional[TutorialProgress]:
        """Get progress for a tutorial."""
        return self._progress.get(tutorial_title)

    def reset(self, tutorial_title: str) -> bool:
        """
        Reset tutorial progress.

        Args:
            tutorial_title: Tutorial to reset

        Returns:
            True if reset
        """
        if tutorial_title in self._progress:
            del self._progress[tutorial_title]
            return True
        return False


def create_vox_tutorials() -> TutorialRunner:
    """
    Create VØX tutorial runner with standard tutorials.

    Returns:
        Configured tutorial runner
    """
    runner = TutorialRunner()

    # ========================================================================
    # Getting Started Tutorial
    # ========================================================================

    getting_started = (
        TutorialBuilder(
            "Getting Started with VØX",
            "Learn the basics of AXIØM VØX text-to-speech",
            TutorialLevel.BEGINNER,
        )
        .set_time(15)
        .add_prerequisite("Python 3.9+")
        .add_prerequisite("Basic Python knowledge")
        .add_tag("beginner")
        .add_tag("intro")
        .add_step(
            title="Installation",
            content="""
First, let's make sure VØX is installed correctly.

VØX is part of the AXIØM kernel and can be imported directly.
""",
            code="""
from axiom_vox import __version__
print(f"VØX Version: {__version__}")
""",
            expected_result="You should see the VØX version number.",
            checkpoint="import_success",
        )
        .add_step(
            title="Voice Space Director",
            content="""
The VoiceSpaceDirector is the heart of VØX. It matches content
to the optimal voice using 8-dimensional voice space analysis.
""",
            code="""
from axiom_vox import VoiceSpaceDirector

director = VoiceSpaceDirector()
result = director.direct(
    text="Welcome to our platform!",
    context={"domain": "greeting"}
)

print(f"Matched voice: {result['matched_voice_id']}")
print(f"Confidence: {result['confidence']:.2f}")
""",
            expected_result="A voice ID and confidence score.",
            hints=["Try different text to see how voice matching changes"],
        )
        .add_step(
            title="Emotion Presets",
            content="""
VØX includes 18 emotion presets for controlling voice tone.
""",
            code="""
from axiom_vox import list_emotion_presets, get_emotion_preset

# See all available presets
presets = list_emotion_presets()
print(f"Available: {presets[:5]}...")

# Get a specific preset
joy = get_emotion_preset("joy")
print(f"Joy preset - pitch: {joy.pitch_shift}, rate: {joy.rate}")
""",
            expected_result="List of presets and joy preset details.",
        )
        .add_step(
            title="SSML Support",
            content="""
VØX supports W3C SSML 1.1 for fine-grained speech control.
""",
            code="""
from axiom_vox import SSMLParser, SSMLGenerator

# Parse SSML
parser = SSMLParser()
doc = parser.parse('<speak>Hello <break time="500ms"/> World</speak>')

# Generate SSML
generator = SSMLGenerator()
generator.add_text("Hello")
generator.add_break("500ms")
generator.add_text("World")
ssml = generator.generate()

print(f"Generated: {ssml[:60]}...")
""",
            expected_result="SSML document string.",
        )
        .add_step(
            title="Congratulations!",
            content="""
You've completed the Getting Started tutorial!

Next steps:
- Try the Voice Matching tutorial for deeper understanding
- Explore the Multi-Voice tutorial for dialogue synthesis
- Check the SDK tutorial for API integration
""",
            code="print('Tutorial complete!')",
            expected_result="Tutorial complete message.",
        )
        .build()
    )

    runner.register(getting_started)

    # ========================================================================
    # Voice Matching Tutorial
    # ========================================================================

    voice_matching = (
        TutorialBuilder(
            "Voice Matching Deep Dive",
            "Understand VØX's 8-dimensional voice space",
            TutorialLevel.INTERMEDIATE,
        )
        .set_time(25)
        .add_prerequisite("Getting Started tutorial")
        .add_tag("voice")
        .add_tag("intermediate")
        .add_step(
            title="Voice Dimensions",
            content="""
VØX uses 8 dimensions to characterize voices:

1. **Formality**: casual ↔ formal
2. **Temperature**: cool ↔ warm
3. **Energy**: calm ↔ energetic
4. **Authority**: peer ↔ authoritative
5. **Pace**: slow ↔ fast
6. **Pitch**: low ↔ high
7. **Resonance**: thin ↔ rich
8. **Age**: young ↔ mature
""",
            code="""
from axiom_vox import VoiceVector

# Create a custom voice vector
vector = VoiceVector(
    formality=0.8,    # Very formal
    temperature=0.7,  # Warm
    energy=0.3,       # Calm
    authority=0.9,    # Authoritative
    pace=0.5,         # Medium
    pitch=0.4,        # Slightly low
    resonance=0.7,    # Rich
    age=0.6,          # Mature
)

print(f"Vector: {vector}")
""",
            expected_result="Voice vector with 8 dimensions.",
        )
        .add_step(
            title="Content Analysis",
            content="""
VØX analyzes text content to determine optimal voice characteristics.
""",
            code="""
from axiom_vox import VoiceSpaceDirector

director = VoiceSpaceDirector()

# Analyze different content types
texts = [
    "Breaking news: major earthquake hits...",
    "Hey! Want to grab coffee later?",
    "The quarterly results show a 15% increase...",
]

for text in texts:
    result = director.direct(text)
    print(f"'{text[:30]}...' -> {result['matched_voice_id']}")
""",
            expected_result="Different voices matched to different content.",
        )
        .add_step(
            title="Voice Profiles",
            content="""
Voice profiles define the characteristics of available voices.
""",
            code="""
from axiom_vox import VoiceProfile

# Create a voice profile
profile = VoiceProfile(
    voice_id="narrator",
    name="Story Narrator",
    vector=VoiceVector(
        formality=0.6,
        temperature=0.8,
        energy=0.4,
        authority=0.5,
    ),
    description="Warm, engaging storytelling voice",
    tags=["narrative", "audiobook"],
)

print(f"Profile: {profile.name}")
print(f"Tags: {profile.tags}")
""",
            expected_result="Voice profile with name and tags.",
            hints=["Profiles can be registered in the voice registry"],
        )
        .build()
    )

    runner.register(voice_matching)

    # ========================================================================
    # SDK Integration Tutorial
    # ========================================================================

    sdk_tutorial = (
        TutorialBuilder(
            "SDK Integration",
            "Integrate VØX into your applications",
            TutorialLevel.INTERMEDIATE,
        )
        .set_time(30)
        .add_prerequisite("Getting Started tutorial")
        .add_tag("sdk")
        .add_tag("integration")
        .add_step(
            title="VoxClient Setup",
            content="""
The VoxClient provides a high-level interface for VØX integration.
""",
            code="""
from axiom_vox import VoxClient, VoxConfig, Environment

# Configure client
config = VoxConfig(
    api_key="your_api_key",
    environment=Environment.DEVELOPMENT,
    timeout_seconds=30,
)

print(f"Environment: {config.environment.value}")
print(f"Timeout: {config.timeout_seconds}s")
""",
            expected_result="Configuration details.",
        )
        .add_step(
            title="Retry Policies",
            content="""
VØX SDK includes automatic retry with exponential backoff.
""",
            code="""
from axiom_vox import RetryConfig, RetryPolicy

# Configure retry behavior
retry_config = RetryConfig(
    max_attempts=3,
    base_delay_ms=100,
    max_delay_ms=5000,
    exponential_base=2.0,
)

policy = RetryPolicy(retry_config)
print(f"Max attempts: {policy.config.max_attempts}")
print(f"Base delay: {policy.config.base_delay_ms}ms")
""",
            expected_result="Retry configuration details.",
        )
        .add_step(
            title="Session Management",
            content="""
Sessions track requests for monitoring and debugging.
""",
            code="""
from axiom_vox import VoxSession, RequestContext

# Create a session
session = VoxSession(session_id="user_123")

# Create request context
context = RequestContext(
    request_id="req_abc",
    session_id=session.session_id,
    user_id="user_123",
)

print(f"Session: {session.session_id}")
print(f"Request: {context.request_id}")
""",
            expected_result="Session and request IDs.",
        )
        .add_step(
            title="Workflows",
            content="""
Use workflow helpers for common patterns.
""",
            code="""
from axiom_vox import WorkflowBuilder, WorkflowStep

# Build a synthesis workflow
workflow = (
    WorkflowBuilder("synthesis_flow")
    .add_step(
        name="validate",
        description="Validate input text",
    )
    .add_step(
        name="select_voice",
        description="Match optimal voice",
    )
    .add_step(
        name="synthesize",
        description="Generate audio",
    )
    .add_step(
        name="quality_check",
        description="Validate output quality",
    )
    .build()
)

print(f"Workflow: {workflow.name}")
print(f"Steps: {len(workflow.steps)}")
""",
            expected_result="Workflow name and step count.",
        )
        .build()
    )

    runner.register(sdk_tutorial)

    # ========================================================================
    # Performance Optimization Tutorial
    # ========================================================================

    performance_tutorial = (
        TutorialBuilder(
            "Performance Optimization",
            "Optimize VØX for production workloads",
            TutorialLevel.ADVANCED,
        )
        .set_time(45)
        .add_prerequisite("SDK Integration tutorial")
        .add_tag("performance")
        .add_tag("advanced")
        .add_step(
            title="Audio Caching",
            content="""
Cache synthesized audio to avoid redundant processing.
""",
            code="""
from axiom_vox import AudioCache

# Create cache
cache = AudioCache(
    max_entries=1000,
    max_bytes=100 * 1024 * 1024,  # 100MB
)

# Cache audio
cache.put("Hello world", "warm", b"audio_data_here")

# Check cache
audio = cache.get("Hello world", "warm")
print(f"Cache hit: {audio is not None}")

# Get stats
stats = cache.get_stats()
print(f"Items: {stats.items}, Hits: {stats.hits}")
""",
            expected_result="Cache hit confirmation and stats.",
        )
        .add_step(
            title="Connection Pooling",
            content="""
Use connection pools for efficient HTTP/WebSocket connections.
""",
            code="""
from axiom_vox import HTTPConnectionPool

# Create connection pool
pool = HTTPConnectionPool(
    base_url="http://localhost:8000",
    pool_size=10,
    timeout=30,
)

# Get stats
stats = pool.get_stats()
print(f"Pool size: {pool.pool_size}")
print(f"Total requests: {stats.total_requests}")
""",
            expected_result="Pool configuration and stats.",
        )
        .add_step(
            title="Batch Processing",
            content="""
Process multiple requests efficiently with batching.
""",
            code="""
import asyncio
from axiom_vox import BatchOptimizer, BatchStrategy

async def demo():
    optimizer = BatchOptimizer(
        max_concurrent=10,
        strategy=BatchStrategy.PARALLEL,
    )

    items = list(range(10))

    async def process(item):
        return item * 2

    result = await optimizer.process(items, process)
    print(f"Completed: {result.stats.completed_items}")
    print(f"Failed: {result.stats.failed_items}")

asyncio.run(demo())
""",
            expected_result="Batch processing stats.",
        )
        .add_step(
            title="Lazy Loading",
            content="""
Load heavy resources on demand.
""",
            code="""
from axiom_vox import LazyLoader

# Create lazy loader
loader = LazyLoader(
    max_loaded=5,
    auto_unload=True,
)

# Register resource
loader.register(
    "heavy_model",
    lambda: {"model": "loaded"},
    memory_estimate_bytes=100 * 1024 * 1024,
)

# Check stats
stats = loader.get_stats()
print(f"Registered resources: {len(loader._resources)}")
print(f"Currently loaded: {stats.current_loaded}")
""",
            expected_result="Resource loader stats.",
        )
        .build()
    )

    runner.register(performance_tutorial)

    return runner

#!/usr/bin/env python3
"""
VOICEPRINT PROFILE TRACER - Core Implementation
═══════════════════════════════════════════════════════════════
Temporal voice evolution analysis system.

This module implements the core functionality for analyzing
how a creator's voice evolves through time.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContentPiece:
    """A single piece of content with metadata"""
    text: str
    date: datetime
    platform: str
    content_type: str
    url: Optional[str] = None
    engagement: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StylisticFeatures:
    """Stylistic features extracted from content"""
    avg_sentence_length: float
    vocab_diversity: float
    punctuation_density: float
    metaphor_count: int
    question_ratio: float
    command_ratio: float
    exclamation_ratio: float
    i_ratio: float  # First person
    you_ratio: float  # Second person
    we_ratio: float  # First person plural
    they_ratio: float  # Third person


@dataclass
class EmotionalTone:
    """Emotional tone analysis"""
    positivity: float  # 0.0 to 1.0
    negativity: float  # 0.0 to 1.0
    anger: float
    joy: float
    awe: float
    sadness: float
    fear: float
    surprise: float
    disgust: float
    trust: float


@dataclass
class TopicDistribution:
    """Topic distribution for a piece of content"""
    topics: Dict[str, float]  # topic -> weight
    dominant_topic: str
    topic_diversity: float


@dataclass
class VoiceEra:
    """A temporal era in a creator's voice evolution"""
    name: str
    start_date: datetime
    end_date: datetime
    dominant_topics: List[str]
    stylistic_traits: StylisticFeatures
    emotional_tone: EmotionalTone
    perspective: Dict[str, float]  # I/you/we/they ratios
    mode: str  # teacher/confessor/commander/entertainer
    description: str
    greatest_hits: List[ContentPiece] = field(default_factory=list)


@dataclass
class VoiceprintSignature:
    """A voiceprint signature (radar chart data)"""
    sentence_rhythm: float  # 0.0 (choppy) to 1.0 (flowing)
    abstraction_level: float  # 0.0 (concrete) to 1.0 (conceptual)
    metaphor_density: float  # 0.0 (sparse) to 1.0 (heavy)
    emotional_palette: Dict[str, float]  # emotion -> intensity
    perspective: Dict[str, float]  # I/you/we/they
    mode: str  # teacher/confessor/commander/entertainer
    era: Optional[str] = None


@dataclass
class ShiftPoint:
    """A point where voice changed significantly"""
    date: datetime
    shift_type: str  # style_change, emotion_shift, topic_shift, perspective_shift
    magnitude: float  # 0.0 to 1.0
    description: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]


@dataclass
class AxiomTriadMap:
    """AXIØM triad mapping for voice"""
    energy: Dict[str, Any]  # Emotional tone, intensity, volatility
    form: Dict[str, Any]  # Structure, rhythm, complexity
    consciousness: Dict[str, Any]  # Purpose, aim, transcendence


@dataclass
class VoiceAnalysis:
    """Complete voice analysis result"""
    creator: str
    content_pieces: List[ContentPiece]
    eras: List[VoiceEra]
    voiceprint_signature: VoiceprintSignature
    shift_points: List[ShiftPoint]
    axiom_triad_map: AxiomTriadMap
    timeline: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════
# VOICEPRINT AGENT
# ═══════════════════════════════════════════════════════════════

class VoiceprintAgent:
    """
    VOICEPRINT Profile Tracer Agent
    
    Traces how a creator's voice evolves through time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VOICEPRINT agent"""
        self.config = config or {}
        self.content_cache: Dict[str, List[ContentPiece]] = {}
        self.analysis_cache: Dict[str, VoiceAnalysis] = {}
        
    # ═══════════════════════════════════════════════════════════
    # CORE FUNCTIONS
    # ═══════════════════════════════════════════════════════════
    
    def ingest_multi_source_content(
        self,
        creator: str,
        sources: Dict[str, Any],
        date_range: Optional[Tuple[str, str]] = None,
        verbosity: int = 10
    ) -> List[ContentPiece]:
        """
        Ingest content from multiple sources with comprehensive historical retrieval.
        
        Uses platform-aware sourcing, Wayback Machine, interviews, transcripts.
        10/10 verbosity search - leaves no stone unturned.
        
        Args:
            creator: Creator identifier (handle, name, etc.)
            sources: Dict mapping platform -> identifier
                Example: {"twitter": "@naval", "youtube": "Naval", "blog": "nav.al"}
            date_range: Optional (start_date, end_date) tuple as strings
            verbosity: 1-10, 10 = maximum thoroughness (default: 10)
            
        Returns:
            List of ContentPiece objects
        """
        import asyncio
        from .historical_ingestion import HistoricalContentIngester
        
        # Parse date range
        start_date = datetime(2010, 1, 1)  # Default
        end_date = datetime.utcnow()
        
        if date_range:
            try:
                start_date = datetime.fromisoformat(date_range[0])
                end_date = datetime.fromisoformat(date_range[1])
            except:
                pass
        
        # Use historical ingester for comprehensive retrieval
        ingester = HistoricalContentIngester(config=self.config)
        
        # Run async ingestion
        async def _ingest():
            raw_content = await ingester.ingest_creator_comprehensive(
                creator=creator,
                identifiers=sources,
                date_range=(start_date, end_date),
                verbosity=verbosity
            )
            await ingester.close()
            return raw_content
        
        raw_content = asyncio.run(_ingest())
        
        # Convert to ContentPiece objects
        pieces = []
        for item in raw_content:
            pieces.append(ContentPiece(
                text=item.get("text", ""),
                date=item.get("date", datetime.utcnow()),
                platform=item.get("platform", "unknown"),
                content_type=item.get("content_type", "unknown"),
                url=item.get("url"),
                engagement=item.get("engagement"),
                metadata=item.get("metadata", {})
            ))
        
        return pieces
    
    def extract_stylistic_features(self, content: ContentPiece) -> StylisticFeatures:
        """
        Extract stylistic features from content.
        
        Args:
            content: ContentPiece to analyze
            
        Returns:
            StylisticFeatures object
        """
        text = content.text
        
        # Basic metrics
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        words = text.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        punctuation_density = punctuation_count / max(len(text), 1)
        
        # Perspective ratios
        i_count = sum(1 for w in words if w in ['i', 'me', 'my', 'mine'])
        you_count = sum(1 for w in words if w in ['you', 'your', 'yours'])
        we_count = sum(1 for w in words if w in ['we', 'us', 'our', 'ours'])
        they_count = sum(1 for w in words if w in ['they', 'them', 'their', 'theirs'])
        total_pronouns = i_count + you_count + we_count + they_count
        
        i_ratio = i_count / max(total_pronouns, 1)
        you_ratio = you_count / max(total_pronouns, 1)
        we_ratio = we_count / max(total_pronouns, 1)
        they_ratio = they_count / max(total_pronouns, 1)
        
        # Question/command ratios
        question_count = text.count('?')
        command_count = sum(1 for s in sentences if any(s.strip().lower().startswith(cmd) 
                                                       for cmd in ['do', 'make', 'get', 'create', 'build']))
        exclamation_count = text.count('!')
        
        total_sentences = len(sentences)
        question_ratio = question_count / max(total_sentences, 1)
        command_ratio = command_count / max(total_sentences, 1)
        exclamation_ratio = exclamation_count / max(total_sentences, 1)
        
        # Metaphor detection (simplified - would use NLP in production)
        metaphor_count = 0  # TODO: Implement metaphor detection
        
        return StylisticFeatures(
            avg_sentence_length=avg_sentence_length,
            vocab_diversity=vocab_diversity,
            punctuation_density=punctuation_density,
            metaphor_count=metaphor_count,
            question_ratio=question_ratio,
            command_ratio=command_ratio,
            exclamation_ratio=exclamation_ratio,
            i_ratio=i_ratio,
            you_ratio=you_ratio,
            we_ratio=we_ratio,
            they_ratio=they_ratio
        )
    
    def analyze_emotional_tone(self, content: ContentPiece) -> EmotionalTone:
        """
        Analyze emotional tone of content.
        
        Args:
            content: ContentPiece to analyze
            
        Returns:
            EmotionalTone object
        """
        # TODO: Implement actual sentiment/emotion analysis
        # Would use NLP models like VADER, TextBlob, or transformer models
        
        # Placeholder implementation
        return EmotionalTone(
            positivity=0.5,
            negativity=0.3,
            anger=0.1,
            joy=0.4,
            awe=0.2,
            sadness=0.1,
            fear=0.1,
            surprise=0.1,
            disgust=0.05,
            trust=0.3
        )
    
    def detect_topic_evolution(
        self,
        content_pieces: List[ContentPiece],
        time_window_days: int = 90
    ) -> Dict[datetime, TopicDistribution]:
        """
        Detect topic evolution over time.
        
        Args:
            content_pieces: List of content pieces
            time_window_days: Window size for topic analysis
            
        Returns:
            Dict mapping time -> TopicDistribution
        """
        # TODO: Implement topic modeling (LDA, BERTopic, etc.)
        # Would cluster content by time windows and extract topics
        
        topic_evolution = {}
        return topic_evolution
    
    def map_perspective_shifts(
        self,
        content_pieces: List[ContentPiece]
    ) -> List[Dict[str, Any]]:
        """
        Map perspective shifts over time.
        
        Args:
            content_pieces: List of content pieces
            
        Returns:
            List of perspective shift events
        """
        shifts = []
        
        # Group by time windows
        # Calculate I/you/we/they ratios per window
        # Detect significant changes
        
        return shifts
    
    def cluster_temporal_eras(
        self,
        content_pieces: List[ContentPiece],
        min_era_duration_months: int = 6
    ) -> List[VoiceEra]:
        """
        Cluster content into temporal eras.
        
        Args:
            content_pieces: List of content pieces
            min_era_duration_months: Minimum duration for an era
            
        Returns:
            List of VoiceEra objects
        """
        # TODO: Implement temporal clustering
        # Would use:
        # - Time-based segmentation
        # - Stylistic similarity clustering
        # - Change point detection
        
        eras = []
        return eras
    
    def generate_voiceprint_signature(
        self,
        content_pieces: List[ContentPiece],
        era: Optional[str] = None
    ) -> VoiceprintSignature:
        """
        Generate voiceprint signature (radar chart data).
        
        Args:
            content_pieces: List of content pieces
            era: Optional era name to filter by
            
        Returns:
            VoiceprintSignature object
        """
        # Aggregate stylistic features
        all_features = [self.extract_stylistic_features(p) for p in content_pieces]
        
        # Calculate averages
        avg_sentence_rhythm = sum(f.avg_sentence_length for f in all_features) / max(len(all_features), 1)
        avg_abstraction = sum(f.vocab_diversity for f in all_features) / max(len(all_features), 1)
        avg_metaphor = sum(f.metaphor_count for f in all_features) / max(len(all_features), 1)
        
        # Normalize to 0-1 scale
        sentence_rhythm = min(avg_sentence_rhythm / 30.0, 1.0)  # Normalize to ~30 words
        abstraction_level = avg_abstraction  # Already 0-1
        metaphor_density = min(avg_metaphor / 10.0, 1.0)  # Normalize to ~10 metaphors
        
        # Aggregate emotional tone
        all_tones = [self.analyze_emotional_tone(p) for p in content_pieces]
        emotional_palette = {
            "anger": sum(t.anger for t in all_tones) / max(len(all_tones), 1),
            "joy": sum(t.joy for t in all_tones) / max(len(all_tones), 1),
            "awe": sum(t.awe for t in all_tones) / max(len(all_tones), 1),
            "sadness": sum(t.sadness for t in all_tones) / max(len(all_tones), 1),
        }
        
        # Aggregate perspective
        perspective = {
            "I": sum(f.i_ratio for f in all_features) / max(len(all_features), 1),
            "you": sum(f.you_ratio for f in all_features) / max(len(all_features), 1),
            "we": sum(f.we_ratio for f in all_features) / max(len(all_features), 1),
            "they": sum(f.they_ratio for f in all_features) / max(len(all_features), 1),
        }
        
        # Determine mode (simplified)
        if perspective["you"] > 0.4:
            mode = "teacher"
        elif perspective["I"] > 0.5:
            mode = "confessor"
        elif perspective["we"] > 0.4:
            mode = "commander"
        else:
            mode = "entertainer"
        
        return VoiceprintSignature(
            sentence_rhythm=sentence_rhythm,
            abstraction_level=abstraction_level,
            metaphor_density=metaphor_density,
            emotional_palette=emotional_palette,
            perspective=perspective,
            mode=mode,
            era=era
        )
    
    def identify_shift_points(
        self,
        content_pieces: List[ContentPiece],
        sensitivity: float = 0.7
    ) -> List[ShiftPoint]:
        """
        Identify points where voice changed significantly.
        
        Args:
            content_pieces: List of content pieces
            sensitivity: Sensitivity threshold (0.0 to 1.0)
            
        Returns:
            List of ShiftPoint objects
        """
        # TODO: Implement change point detection
        # Would use:
        # - Statistical change point detection
        # - Sliding window comparison
        # - Multi-dimensional shift detection
        
        shift_points = []
        return shift_points
    
    def create_voice_timeline(
        self,
        eras: List[VoiceEra]
    ) -> Dict[str, Any]:
        """
        Create interactive voice timeline.
        
        Args:
            eras: List of VoiceEra objects
            
        Returns:
            Timeline data structure
        """
        timeline = {
            "eras": [
                {
                    "name": era.name,
                    "start": era.start_date.isoformat(),
                    "end": era.end_date.isoformat(),
                    "topics": era.dominant_topics,
                    "style": {
                        "sentence_length": era.stylistic_traits.avg_sentence_length,
                        "vocab_diversity": era.stylistic_traits.vocab_diversity,
                    },
                    "emotion": {
                        "positivity": era.emotional_tone.positivity,
                        "dominant_emotion": max(
                            ["anger", "joy", "awe", "sadness"],
                            key=lambda e: getattr(era.emotional_tone, e)
                        )
                    },
                    "description": era.description
                }
                for era in eras
            ]
        }
        
        return timeline
    
    def map_axiom_triad(
        self,
        content_pieces: List[ContentPiece],
        era: Optional[str] = None
    ) -> AxiomTriadMap:
        """
        Map voice through AXIØM triad (Energy/Form/Consciousness).
        
        Args:
            content_pieces: List of content pieces
            era: Optional era name to filter by
            
        Returns:
            AxiomTriadMap object
        """
        # Aggregate features
        all_features = [self.extract_stylistic_features(p) for p in content_pieces]
        all_tones = [self.analyze_emotional_tone(p) for p in content_pieces]
        
        # Energy (Field): Emotional tone, intensity, volatility
        energy = {
            "emotional_tone": {
                "positivity": sum(t.positivity for t in all_tones) / max(len(all_tones), 1),
                "negativity": sum(t.negativity for t in all_tones) / max(len(all_tones), 1),
            },
            "intensity": sum(t.anger + t.joy + t.awe for t in all_tones) / max(len(all_tones), 1),
            "volatility": 0.5,  # TODO: Calculate from variance
            "hope": sum(t.joy + t.trust for t in all_tones) / max(len(all_tones), 1),
            "despair": sum(t.sadness + t.fear for t in all_tones) / max(len(all_tones), 1),
        }
        
        # Form (Geometry): Structure, rhythm, complexity
        form = {
            "structure": {
                "sentence_rhythm": sum(f.avg_sentence_length for f in all_features) / max(len(all_features), 1),
                "punctuation_density": sum(f.punctuation_density for f in all_features) / max(len(all_features), 1),
            },
            "rhythm": {
                "question_ratio": sum(f.question_ratio for f in all_features) / max(len(all_features), 1),
                "command_ratio": sum(f.command_ratio for f in all_features) / max(len(all_features), 1),
            },
            "complexity": {
                "vocab_diversity": sum(f.vocab_diversity for f in all_features) / max(len(all_features), 1),
                "abstraction_level": sum(f.vocab_diversity for f in all_features) / max(len(all_features), 1),
            }
        }
        
        # Consciousness (Teleology): Purpose, aim, transcendence
        # Infer from patterns
        avg_you_ratio = sum(f.you_ratio for f in all_features) / max(len(all_features), 1)
        avg_i_ratio = sum(f.i_ratio for f in all_features) / max(len(all_features), 1)
        avg_we_ratio = sum(f.we_ratio for f in all_features) / max(len(all_features), 1)
        
        if avg_you_ratio > 0.4:
            aim = "teaching"
        elif avg_i_ratio > 0.5:
            aim = "self_expression"
        elif avg_we_ratio > 0.4:
            aim = "tribe_building"
        else:
            aim = "attention"
        
        consciousness = {
            "purpose": aim,
            "aim": {
                "attention": 0.3,  # TODO: Infer from engagement patterns
                "truth": 0.4,  # TODO: Infer from content type
                "money": 0.2,  # TODO: Infer from monetization signals
                "tribe": avg_we_ratio,
                "transcendence": sum(t.awe for t in all_tones) / max(len(all_tones), 1),
            }
        }
        
        return AxiomTriadMap(
            energy=energy,
            form=form,
            consciousness=consciousness
        )
    
    def generate_era_greatest_hits(
        self,
        era: VoiceEra,
        top_n: int = 10
    ) -> List[ContentPiece]:
        """
        Generate greatest hits for an era.
        
        Args:
            era: VoiceEra object
            top_n: Number of hits to return
            
        Returns:
            List of ContentPiece objects
        """
        # TODO: Implement ranking algorithm
        # Would consider:
        # - Engagement metrics
        # - Stylistic representativeness
        # - Topic representativeness
        # - Emotional representativeness
        
        return era.greatest_hits[:top_n]
    
    def compare_voice_evolution(
        self,
        analysis_a: VoiceAnalysis,
        analysis_b: VoiceAnalysis,
        era_a: Optional[str] = None,
        era_b: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare voice evolution between two analyses.
        
        Args:
            analysis_a: First VoiceAnalysis
            analysis_b: Second VoiceAnalysis (or same creator, different era)
            era_a: Optional era name from analysis_a
            era_b: Optional era name from analysis_b
            
        Returns:
            Comparison data structure
        """
        # Get eras to compare
        era_a_obj = next((e for e in analysis_a.eras if e.name == era_a), None) if era_a else None
        era_b_obj = next((e for e in analysis_b.eras if e.name == era_b), None) if era_b else None
        
        if not era_a_obj:
            era_a_obj = analysis_a.eras[0] if analysis_a.eras else None
        if not era_b_obj:
            era_b_obj = analysis_b.eras[0] if analysis_b.eras else None
        
        if not era_a_obj or not era_b_obj:
            return {"error": "No eras to compare"}
        
        comparison = {
            "creator_a": analysis_a.creator,
            "creator_b": analysis_b.creator,
            "era_a": era_a_obj.name,
            "era_b": era_b_obj.name,
            "stylistic_differences": {
                "sentence_length": era_a_obj.stylistic_traits.avg_sentence_length - era_b_obj.stylistic_traits.avg_sentence_length,
                "vocab_diversity": era_a_obj.stylistic_traits.vocab_diversity - era_b_obj.stylistic_traits.vocab_diversity,
            },
            "emotional_differences": {
                "positivity": era_a_obj.emotional_tone.positivity - era_b_obj.emotional_tone.positivity,
                "joy": era_a_obj.emotional_tone.joy - era_b_obj.emotional_tone.joy,
            },
            "perspective_differences": {
                "i_ratio": era_a_obj.perspective.get("I", 0) - era_b_obj.perspective.get("I", 0),
                "you_ratio": era_a_obj.perspective.get("you", 0) - era_b_obj.perspective.get("you", 0),
                "we_ratio": era_a_obj.perspective.get("we", 0) - era_b_obj.perspective.get("we", 0),
            }
        }
        
        return comparison
    
    # ═══════════════════════════════════════════════════════════
    # HIGH-LEVEL API
    # ═══════════════════════════════════════════════════════════
    
    def analyze_creator(
        self,
        creator: str,
        sources: Dict[str, Any],
        date_range: Optional[Tuple[str, str]] = None,
        min_era_duration_months: int = 6
    ) -> VoiceAnalysis:
        """
        Complete voice analysis for a creator.
        
        Args:
            creator: Creator identifier
            sources: Dict mapping platform -> identifier
            date_range: Optional (start_date, end_date) tuple
            min_era_duration_months: Minimum era duration
            
        Returns:
            VoiceAnalysis object
        """
        # Check cache
        cache_key = f"{creator}_{date_range}_{min_era_duration_months}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Ingest content
        content_pieces = self.ingest_multi_source_content(creator, sources, date_range)
        
        # Cluster eras
        eras = self.cluster_temporal_eras(content_pieces, min_era_duration_months)
        
        # Generate voiceprint signature
        voiceprint = self.generate_voiceprint_signature(content_pieces)
        
        # Identify shift points
        shift_points = self.identify_shift_points(content_pieces)
        
        # Map AXIØM triad
        axiom_map = self.map_axiom_triad(content_pieces)
        
        # Create timeline
        timeline = self.create_voice_timeline(eras)
        
        # Create analysis
        analysis = VoiceAnalysis(
            creator=creator,
            content_pieces=content_pieces,
            eras=eras,
            voiceprint_signature=voiceprint,
            shift_points=shift_points,
            axiom_triad_map=axiom_map,
            timeline=timeline
        )
        
        # Cache
        self.analysis_cache[cache_key] = analysis
        
        return analysis


# ═══════════════════════════════════════════════════════════════
# INSTANCE
# ═══════════════════════════════════════════════════════════════

# Load agent instance configuration
_instance_config = None
_instance_path = Path(__file__).parent / "VOICEPRINT_instance.yaml"

if _instance_path.exists():
    with open(_instance_path, 'r') as f:
        _instance_config = yaml.safe_load(f)

# Create agent instance
VOICEPRINT_instance = VoiceprintAgent(config=_instance_config)




















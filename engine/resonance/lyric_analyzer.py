"""
VØX Resonance - Lyric Analyzer
------------------------------

Extracts lyric features relevant to psychological resonance.

Features:
    - Sentiment analysis
    - Theme detection
    - Pronoun analysis (I/you/we)
    - Repetition patterns
    - Imagery and abstraction
    - Temporal focus (past/present/future)
    - Emotional word detection
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set, Tuple

from .models import LyricFeatures, EmotionalValence

logger = logging.getLogger(__name__)


# Emotional word lexicons
POSITIVE_WORDS = {
    "love", "happy", "joy", "beautiful", "wonderful", "amazing", "great",
    "good", "best", "blessed", "bright", "free", "hope", "dream", "smile",
    "light", "peace", "warm", "sweet", "gentle", "kind", "true", "alive",
    "strong", "rise", "fly", "shine", "heaven", "angel", "together", "forever",
    "believe", "trust", "faith", "grateful", "magic", "perfect", "paradise",
}

NEGATIVE_WORDS = {
    "hate", "sad", "pain", "hurt", "broken", "lost", "alone", "dark",
    "dead", "death", "die", "cry", "tears", "fear", "scared", "angry",
    "wrong", "bad", "worst", "hell", "devil", "cold", "empty", "nothing",
    "never", "gone", "fall", "drown", "burn", "bleed", "suffer", "regret",
    "betrayed", "lies", "fake", "worthless", "hopeless", "desperate",
}

THEME_KEYWORDS = {
    "love": {"love", "heart", "kiss", "hold", "touch", "romance", "baby", "darling"},
    "loss": {"lost", "gone", "miss", "goodbye", "leave", "left", "forget", "memory"},
    "rebellion": {"fight", "rebel", "riot", "revolution", "against", "break", "rage"},
    "party": {"dance", "party", "club", "night", "drink", "fun", "wild", "crazy"},
    "self-empowerment": {"strong", "power", "rise", "stand", "believe", "can", "will"},
    "spirituality": {"god", "heaven", "soul", "spirit", "pray", "faith", "divine"},
    "nature": {"sun", "moon", "stars", "ocean", "river", "mountain", "sky", "earth"},
    "loneliness": {"alone", "lonely", "empty", "nobody", "silence", "cold", "dark"},
    "desire": {"want", "need", "crave", "hunger", "thirst", "desire", "lust"},
    "nostalgia": {"remember", "yesterday", "used to", "back then", "once", "before"},
    "hope": {"hope", "dream", "tomorrow", "someday", "believe", "wish", "future"},
    "anger": {"hate", "angry", "rage", "fire", "burn", "destroy", "revenge"},
    "freedom": {"free", "fly", "escape", "run", "break free", "liberate", "release"},
    "identity": {"who am i", "myself", "identity", "real", "true self", "inside"},
}

PAST_MARKERS = {"was", "were", "had", "did", "used to", "remember", "once", "before", "yesterday"}
PRESENT_MARKERS = {"is", "am", "are", "now", "today", "here", "being", "feeling"}
FUTURE_MARKERS = {"will", "gonna", "going to", "tomorrow", "someday", "soon", "future"}

IMAGERY_WORDS = {
    "red", "blue", "green", "gold", "silver", "black", "white", "bright",
    "dark", "fire", "ice", "water", "rain", "snow", "sun", "moon", "stars",
    "ocean", "mountain", "river", "forest", "sky", "earth", "wind", "storm",
    "blood", "rose", "diamond", "crystal", "shadow", "light", "smoke", "glass",
}


@dataclass
class LyricAnalyzerConfig:
    """
    Configuration for lyric analyzer.

    Attributes:
        detect_themes: Enable theme detection
        detect_imagery: Enable imagery analysis
        min_repetition_count: Minimum count to consider as repetition
    """
    detect_themes: bool = True
    detect_imagery: bool = True
    min_repetition_count: int = 3


class LyricAnalyzer:
    """
    Analyzer for extracting resonance-relevant lyric features.

    Works without external NLP libraries using lexicon-based analysis.
    """

    def __init__(
        self,
        config: Optional[LyricAnalyzerConfig] = None,
    ):
        """
        Initialize lyric analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or LyricAnalyzerConfig()

    def analyze(self, lyrics: str) -> LyricFeatures:
        """
        Analyze lyrics and extract features.

        Args:
            lyrics: Song lyrics text

        Returns:
            Extracted lyric features
        """
        features = LyricFeatures(raw_text=lyrics)

        # Clean and tokenize
        words = self._tokenize(lyrics)
        lines = [l.strip() for l in lyrics.split('\n') if l.strip()]

        # Basic counts
        features.word_count = len(words)
        features.unique_words = len(set(words))

        if features.word_count > 0:
            features.vocabulary_richness = features.unique_words / features.word_count
        else:
            features.vocabulary_richness = 0

        # Sentiment analysis
        features.sentiment_score = self._analyze_sentiment(words)
        features.emotional_valence = self._score_to_valence(features.sentiment_score)

        # Theme detection
        if self.config.detect_themes:
            features.themes = self._detect_themes(words)

        # Pronoun analysis
        pronoun_analysis = self._analyze_pronouns(words)
        features.first_person_ratio = pronoun_analysis["first_person"]
        features.second_person_ratio = pronoun_analysis["second_person"]

        # Imperative and questions
        features.imperative_count = self._count_imperatives(lines)
        features.question_count = self._count_questions(lines)

        # Repetition analysis
        repetition = self._analyze_repetition(lines)
        features.repetition_score = repetition["score"]
        features.most_repeated = repetition["phrases"]

        # Imagery
        if self.config.detect_imagery:
            features.imagery_score = self._analyze_imagery(words)

        # Abstraction
        features.abstraction_score = self._analyze_abstraction(words)

        # Temporal focus
        features.temporal_focus = self._analyze_temporal_focus(words)

        # Negation
        features.negation_ratio = self._analyze_negation(words)

        # Emotional words
        features.emotional_words = self._extract_emotional_words(words)

        return features

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z\']+\b', text)
        return words

    def _analyze_sentiment(self, words: List[str]) -> float:
        """Calculate sentiment score (-1 to 1)."""
        positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)

        total_emotional = positive_count + negative_count
        if total_emotional == 0:
            return 0.0

        # Calculate weighted score
        score = (positive_count - negative_count) / total_emotional

        # Dampen by proportion of emotional words
        emotional_ratio = total_emotional / max(len(words), 1)
        score = score * min(emotional_ratio * 5, 1)  # Scale up small emotional content

        return max(-1.0, min(1.0, score))

    def _score_to_valence(self, score: float) -> EmotionalValence:
        """Convert sentiment score to valence category."""
        if score < -0.5:
            return EmotionalValence.VERY_NEGATIVE
        elif score < -0.15:
            return EmotionalValence.NEGATIVE
        elif score < 0.15:
            return EmotionalValence.NEUTRAL
        elif score < 0.5:
            return EmotionalValence.POSITIVE
        else:
            return EmotionalValence.VERY_POSITIVE

    def _detect_themes(self, words: List[str]) -> Dict[str, float]:
        """Detect themes with confidence scores."""
        word_set = set(words)
        themes = {}

        for theme, keywords in THEME_KEYWORDS.items():
            matches = word_set & keywords
            if matches:
                # Confidence based on number of matches and frequency
                match_count = sum(words.count(m) for m in matches)
                confidence = min(1.0, match_count / 10)  # Cap at 1.0
                if confidence > 0.1:
                    themes[theme] = round(confidence, 2)

        # Sort by confidence
        themes = dict(sorted(themes.items(), key=lambda x: -x[1]))
        return themes

    def _analyze_pronouns(self, words: List[str]) -> Dict[str, float]:
        """Analyze pronoun usage."""
        first_person = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
        second_person = {"you", "your", "yours", "yourself"}
        third_person = {"he", "she", "they", "him", "her", "them", "his", "hers", "their"}

        total = len(words)
        if total == 0:
            return {"first_person": 0, "second_person": 0, "third_person": 0}

        fp_count = sum(1 for w in words if w in first_person)
        sp_count = sum(1 for w in words if w in second_person)
        tp_count = sum(1 for w in words if w in third_person)

        return {
            "first_person": fp_count / total,
            "second_person": sp_count / total,
            "third_person": tp_count / total,
        }

    def _count_imperatives(self, lines: List[str]) -> int:
        """Count imperative sentences (commands)."""
        imperatives = 0
        imperative_starters = {
            "let", "come", "go", "take", "make", "give", "stop", "keep",
            "tell", "show", "bring", "hold", "run", "stand", "rise",
            "dance", "sing", "listen", "look", "watch", "wait", "stay",
            "don't", "never", "always", "just", "feel", "be",
        }

        for line in lines:
            words = line.lower().split()
            if words:
                first_word = words[0].strip('.,!?')
                if first_word in imperative_starters:
                    imperatives += 1

        return imperatives

    def _count_questions(self, lines: List[str]) -> int:
        """Count questions."""
        return sum(1 for line in lines if '?' in line)

    def _analyze_repetition(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze line repetition (hooks, choruses)."""
        # Normalize lines
        normalized = [re.sub(r'[^\w\s]', '', l.lower().strip()) for l in lines]
        normalized = [l for l in normalized if len(l) > 5]  # Skip short lines

        # Count occurrences
        counter = Counter(normalized)

        # Find repeated phrases
        repeated = [
            phrase for phrase, count in counter.most_common(5)
            if count >= self.config.min_repetition_count
        ]

        # Calculate repetition score
        total_lines = len(normalized)
        if total_lines == 0:
            return {"score": 0, "phrases": []}

        repeated_lines = sum(count for phrase, count in counter.items() if count > 1)
        score = repeated_lines / total_lines

        return {
            "score": min(1.0, score),
            "phrases": repeated[:3],  # Top 3 repeated phrases
        }

    def _analyze_imagery(self, words: List[str]) -> float:
        """Analyze vividness of imagery (0-1)."""
        word_set = set(words)
        imagery_count = len(word_set & IMAGERY_WORDS)

        if len(word_set) == 0:
            return 0.5

        # Score based on imagery word density
        score = imagery_count / len(word_set) * 10  # Scale up
        return min(1.0, score)

    def _analyze_abstraction(self, words: List[str]) -> float:
        """Analyze abstraction level (0=concrete, 1=abstract)."""
        # Abstract concepts
        abstract_words = {
            "love", "hate", "truth", "lies", "soul", "spirit", "mind",
            "heart", "dream", "hope", "fear", "time", "life", "death",
            "freedom", "destiny", "fate", "power", "beauty", "pain",
            "peace", "war", "justice", "faith", "belief", "meaning",
        }

        # Concrete words (things you can see/touch)
        concrete_words = {
            "hand", "face", "eyes", "mouth", "body", "car", "house",
            "door", "window", "table", "chair", "phone", "money",
            "street", "road", "room", "bed", "clothes", "food", "drink",
        }

        word_set = set(words)
        abstract_count = len(word_set & abstract_words)
        concrete_count = len(word_set & concrete_words)

        total = abstract_count + concrete_count
        if total == 0:
            return 0.5

        return abstract_count / total

    def _analyze_temporal_focus(self, words: List[str]) -> Dict[str, float]:
        """Analyze past/present/future focus."""
        word_set = set(words)

        past = len(word_set & PAST_MARKERS)
        present = len(word_set & PRESENT_MARKERS)
        future = len(word_set & FUTURE_MARKERS)

        total = past + present + future
        if total == 0:
            return {"past": 0.33, "present": 0.34, "future": 0.33}

        return {
            "past": past / total,
            "present": present / total,
            "future": future / total,
        }

    def _analyze_negation(self, words: List[str]) -> float:
        """Analyze negation frequency."""
        negations = {"not", "no", "never", "nothing", "nobody", "nowhere",
                    "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
                    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"}

        negation_count = sum(1 for w in words if w in negations)
        total = len(words)

        if total == 0:
            return 0

        return negation_count / total

    def _extract_emotional_words(self, words: List[str]) -> List[str]:
        """Extract emotionally charged words."""
        emotional = []

        for word in words:
            if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
                if word not in emotional:
                    emotional.append(word)

        return emotional[:20]  # Limit to top 20


def analyze_lyrics(
    lyrics: str,
    config: Optional[LyricAnalyzerConfig] = None,
) -> LyricFeatures:
    """
    Analyze lyrics for resonance features.

    Args:
        lyrics: Song lyrics text
        config: Optional analyzer config

    Returns:
        Extracted lyric features
    """
    analyzer = LyricAnalyzer(config)
    return analyzer.analyze(lyrics)

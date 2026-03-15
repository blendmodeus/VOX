"""
VØX STT Tests
=============

Comprehensive tests for the governed STT engine.
Tests follow the patterns established in test_governance.py and test_streaming.py.

Run:
    cd /Users/jeremybrasher/Development/axiom-vox/engine
    python -m pytest tests/test_stt.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestSTTModels:
    """Test data models for the STT pipeline."""

    def test_transcription_config_defaults(self):
        from axiom_vox.stt.models import TranscriptionConfig, STTModelSize

        config = TranscriptionConfig()
        assert config.model_size == STTModelSize.BASE
        assert config.language is None  # auto-detect
        assert config.beam_size == 5
        assert config.word_timestamps is True
        assert config.vad_filter is True
        assert config.govern is True

    def test_transcription_config_custom(self):
        from axiom_vox.stt.models import TranscriptionConfig, STTModelSize

        config = TranscriptionConfig(
            model_size=STTModelSize.LARGE_V3,
            language="en",
            beam_size=3,
            word_timestamps=False,
            govern=False,
        )
        assert config.model_size == STTModelSize.LARGE_V3
        assert config.language == "en"
        assert config.govern is False

    def test_transcription_config_to_dict(self):
        from axiom_vox.stt.models import TranscriptionConfig

        config = TranscriptionConfig()
        d = config.to_dict()
        assert d["model_size"] == "base"
        assert d["word_timestamps"] is True
        assert "govern" in d

    def test_word_timestamp(self):
        from axiom_vox.stt.models import WordTimestamp

        w = WordTimestamp(word="hello", start=0.5, end=1.2, probability=0.95)
        assert w.word == "hello"
        d = w.to_dict()
        assert d["start"] == 0.5
        assert d["probability"] == 0.95

    def test_transcription_segment(self):
        from axiom_vox.stt.models import TranscriptionSegment, WordTimestamp

        words = [
            WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.9),
            WordTimestamp(word="world", start=0.5, end=1.0, probability=0.95),
        ]
        seg = TranscriptionSegment(
            id=0,
            text="hello world",
            start=0.0,
            end=1.0,
            confidence=0.92,
            words=words,
        )
        assert seg.duration == 1.0
        d = seg.to_dict()
        assert len(d["words"]) == 2
        assert d["confidence"] == 0.92

    def test_transcription_result(self):
        from axiom_vox.stt.models import TranscriptionResult, TranscriptionSegment

        segments = [
            TranscriptionSegment(id=0, text="Hello", start=0.0, end=0.5, confidence=0.9),
            TranscriptionSegment(id=1, text="world", start=0.5, end=1.0, confidence=0.8),
        ]
        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            duration_audio=1.0,
            duration_processing=0.2,
        )
        assert result.word_count == 2
        assert result.avg_confidence == pytest.approx(0.85)
        assert result.rtf == 0.2  # 0.2/1.0

    def test_transcription_result_empty(self):
        from axiom_vox.stt.models import TranscriptionResult

        result = TranscriptionResult(text="")
        assert result.word_count == 0
        assert result.avg_confidence == 0.0
        assert result.rtf == 0.0

    def test_model_catalog(self):
        from axiom_vox.stt.models import AVAILABLE_MODELS, get_model_info

        assert len(AVAILABLE_MODELS) == 6
        assert AVAILABLE_MODELS[0].name == "tiny"
        assert AVAILABLE_MODELS[-1].name == "distil-large-v3"

        info = get_model_info("base")
        assert info is not None
        assert info.size_mb == 140

        assert get_model_info("nonexistent") is None

    def test_stt_model_size_enum(self):
        from axiom_vox.stt.models import STTModelSize

        assert STTModelSize.TINY.value == "tiny"
        assert STTModelSize.LARGE_V3.value == "large-v3"

    def test_result_to_dict_has_all_fields(self):
        from axiom_vox.stt.models import TranscriptionResult

        result = TranscriptionResult(text="test", language="en")
        d = result.to_dict()
        required_keys = [
            "text", "language", "duration_audio", "duration_processing",
            "rtf", "word_count", "avg_confidence", "model_size",
            "device", "segments", "status",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"
        assert d["status"] == "success"


# ============================================================================
# TRANSCRIBER TESTS
# ============================================================================

class TestVoxTranscriber:
    """Test the VoxTranscriber wrapper."""

    def test_init_defaults(self):
        from axiom_vox.stt.transcriber import VoxTranscriber

        t = VoxTranscriber()
        assert t.model_size == "base"
        assert t.is_loaded is False

    def test_init_custom(self):
        from axiom_vox.stt.transcriber import VoxTranscriber

        t = VoxTranscriber(model_size="small", device="cpu")
        assert t.model_size == "small"
        assert t.device == "cpu"

    def test_get_info(self):
        from axiom_vox.stt.transcriber import VoxTranscriber

        t = VoxTranscriber(model_size="base", device="cpu")
        info = t.get_info()
        assert info["model_size"] == "base"
        assert info["loaded"] is False
        assert "has_faster_whisper" in info

    def test_list_models(self):
        from axiom_vox.stt.transcriber import VoxTranscriber

        models = VoxTranscriber.list_models()
        assert len(models) == 6
        assert models[0].name == "tiny"

    def test_unload_model_when_not_loaded(self):
        from axiom_vox.stt.transcriber import VoxTranscriber

        t = VoxTranscriber()
        t.unload_model()  # Should not raise
        assert t.is_loaded is False

    def test_device_detection(self):
        from axiom_vox.stt.transcriber import detect_device

        device = detect_device()
        assert device in ("cpu", "cuda", "mps")


# ============================================================================
# GOVERNOR TESTS
# ============================================================================

class TestSTTGovernor:
    """Test post-transcription governance pipeline."""

    def _make_result(self, text, segments=None):
        from axiom_vox.stt.models import TranscriptionResult
        return TranscriptionResult(text=text, segments=segments or [])

    def test_init_defaults(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        assert gov.config.redact_pii is True
        assert gov.config.filter_content is True

    def test_clean_text_passes(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("Hello, how are you today?")
        governed = gov.govern(result)

        assert governed.governed_text == "Hello, how are you today?"
        assert governed.is_clean is True
        assert len(governed.redactions) == 0

    def test_email_redaction(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("Contact me at john@example.com please")
        governed = gov.govern(result)

        assert "john@example.com" not in governed.governed_text
        assert "[REDACTED]" in governed.governed_text
        assert governed.has_redactions is True
        assert any(r.type.value == "email" for r in governed.redactions)

    def test_phone_redaction(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("Call me at 555-123-4567")
        governed = gov.govern(result)

        assert "555-123-4567" not in governed.governed_text
        assert "[REDACTED]" in governed.governed_text
        assert any(r.type.value == "phone" for r in governed.redactions)

    def test_ssn_redaction(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("My SSN is 123-45-6789")
        governed = gov.govern(result)

        assert "123-45-6789" not in governed.governed_text
        assert "[REDACTED]" in governed.governed_text

    def test_credit_card_redaction(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("Card number 4111-1111-1111-1111")
        governed = gov.govern(result)

        assert "4111-1111-1111-1111" not in governed.governed_text
        assert "[REDACTED]" in governed.governed_text

    def test_multiple_pii_redaction(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("Email john@test.com, call 555-987-6543")
        governed = gov.govern(result)

        assert "john@test.com" not in governed.governed_text
        assert "555-987-6543" not in governed.governed_text
        assert governed.governed_text.count("[REDACTED]") >= 2

    def test_pii_redaction_disabled(self):
        from axiom_vox.stt.governor import STTGovernor, STTGovernanceConfig

        config = STTGovernanceConfig(redact_pii=False)
        gov = STTGovernor(config=config)
        result = self._make_result("Contact john@example.com")
        governed = gov.govern(result)

        assert "john@example.com" in governed.governed_text

    def test_low_confidence_flagging(self):
        from axiom_vox.stt.governor import STTGovernor
        from axiom_vox.stt.models import TranscriptionResult, TranscriptionSegment

        segments = [
            TranscriptionSegment(id=0, text="Good", start=0, end=1, confidence=0.9),
            TranscriptionSegment(id=1, text="Bad", start=1, end=2, confidence=0.1),
        ]
        result = TranscriptionResult(text="Good Bad", segments=segments)
        gov = STTGovernor()
        governed = gov.govern(result)

        assert any("low_confidence" in f for f in governed.flags)
        assert 1 in governed.segments_flagged

    def test_no_speech_flagging(self):
        from axiom_vox.stt.governor import STTGovernor
        from axiom_vox.stt.models import TranscriptionResult, TranscriptionSegment

        segments = [
            TranscriptionSegment(
                id=0, text="", start=0, end=1,
                confidence=0.5, no_speech_prob=0.95
            ),
        ]
        result = TranscriptionResult(text="", segments=segments)
        gov = STTGovernor()
        governed = gov.govern(result)

        assert any("no_speech" in f for f in governed.flags)

    def test_content_filtering_medical(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("The patient has a diagnosis of...")
        governed = gov.govern(result)

        assert any("sensitive:medical_info" in f for f in governed.flags)

    def test_content_filtering_financial(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("My bank account routing number is...")
        governed = gov.govern(result)

        assert any("sensitive:financial_info" in f for f in governed.flags)

    def test_governance_result_to_dict(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        result = self._make_result("test@email.com and 555-111-2222")
        governed = gov.govern(result)
        d = governed.to_dict()

        assert "governed_text" in d
        assert "original_text" in d
        assert "redactions" in d
        assert "redaction_count" in d
        assert d["governance_applied"] is True

    def test_quick_redact(self):
        from axiom_vox.stt.governor import STTGovernor

        gov = STTGovernor()
        redacted = gov.quick_redact("Contact john@test.com please")
        assert "john@test.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_custom_replacement(self):
        from axiom_vox.stt.governor import STTGovernor, STTGovernanceConfig

        config = STTGovernanceConfig(pii_replacement="***")
        gov = STTGovernor(config=config)
        result = self._make_result("Email: test@test.com")
        governed = gov.govern(result)

        assert "***" in governed.governed_text
        assert "[REDACTED]" not in governed.governed_text


# ============================================================================
# STREAMING TESTS
# ============================================================================

class TestStreamingTranscriber:
    """Test streaming session management."""

    def test_create_session(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()
        assert session.session_id.startswith("stt_stream_")
        assert session.state.value == "pending"

    def test_feed_chunk_advances_state(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()
        st.feed_chunk(session.session_id, b"\x00" * 1000)

        assert session.state.value == "listening"
        assert session.chunks_received == 1
        assert session.bytes_received == 1000

    def test_feed_multiple_chunks(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()

        for i in range(5):
            st.feed_chunk(session.session_id, b"\x00" * 500)

        assert session.chunks_received == 5
        assert session.bytes_received == 2500

    def test_unknown_session_raises(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        with pytest.raises(ValueError, match="Unknown session"):
            st.feed_chunk("nonexistent_id", b"\x00")

    def test_cancel_session(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()
        result = st.cancel_session(session.session_id)

        assert result is True
        assert session.state.value == "cancelled"

    def test_cancel_unknown_session(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        assert st.cancel_session("nonexistent") is False

    def test_remove_session(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()
        st.remove_session(session.session_id)

        assert st.get_session(session.session_id) is None

    def test_session_progress(self):
        from axiom_vox.stt.streaming import StreamingTranscriber

        st = StreamingTranscriber()
        session = st.create_session()
        st.feed_chunk(session.session_id, b"\x00" * 800)

        progress = session.get_progress()
        assert progress["state"] == "listening"
        assert progress["chunks_received"] == 1

    def test_stream_config(self):
        from axiom_vox.stt.streaming import STTStreamConfig

        config = STTStreamConfig(chunk_duration_ms=250, language="en")
        d = config.to_dict()
        assert d["chunk_duration_ms"] == 250
        assert d["language"] == "en"

    def test_stream_message(self):
        from axiom_vox.stt.streaming import STTStreamMessage

        msg = STTStreamMessage(
            type="partial",
            session_id="test_123",
            text="Hello",
        )
        j = msg.to_json()
        assert j["type"] == "partial"
        assert j["text"] == "Hello"
        assert "timestamp" in j


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================

class TestModuleExports:
    """Verify all expected symbols are exported."""

    def test_import_models(self):
        from axiom_vox.stt import (
            TranscriptionConfig,
            TranscriptionResult,
            TranscriptionSegment,
            WordTimestamp,
            STTModelSize,
            STTModelInfo,
            AVAILABLE_MODELS,
        )
        assert len(AVAILABLE_MODELS) > 0

    def test_import_transcriber(self):
        from axiom_vox.stt import VoxTranscriber, get_transcriber, transcribe
        assert callable(transcribe)

    def test_import_governor(self):
        from axiom_vox.stt import (
            STTGovernor,
            STTGovernanceConfig,
            STTGovernanceResult,
            govern_transcription,
        )
        assert callable(govern_transcription)

    def test_import_streaming(self):
        from axiom_vox.stt import (
            StreamingTranscriber,
            STTStreamConfig,
            STTStreamSession,
            STTStreamState,
        )
        assert STTStreamState.LISTENING.value == "listening"

    def test_top_level_import(self):
        """Verify stt is accessible from axiom_vox namespace."""
        import axiom_vox.stt
        assert hasattr(axiom_vox.stt, 'VoxTranscriber')
        assert hasattr(axiom_vox.stt, 'STTGovernor')
        assert hasattr(axiom_vox.stt, 'TextFormatter')

    def test_import_formatter(self):
        from axiom_vox.stt import (
            TextFormatter,
            FormatMode,
            FormatResult,
            get_formatter,
            clean_text,
        )
        assert callable(clean_text)


# ============================================================================
# FORMATTER TESTS
# ============================================================================

class TestTextFormatter:
    """Test AI text formatting — the Glaido killer feature."""

    def test_raw_mode_no_change(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("so um I was like thinking", FormatMode.RAW)
        assert result.formatted_text == "so um I was like thinking"
        assert result.mode == FormatMode.RAW

    def test_empty_input(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("", FormatMode.CLEAN)
        assert result.formatted_text == ""

    def test_whitespace_only(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("   ", FormatMode.CLEAN)
        assert result.formatted_text == ""

    def test_filler_um_removal(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I was um thinking about it", FormatMode.CLEAN)
        assert "um" not in result.formatted_text.lower()
        assert "thinking" in result.formatted_text

    def test_filler_uh_removal(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("uh we should uh go there", FormatMode.CLEAN)
        assert "uh" not in result.formatted_text.lower().split()

    def test_filler_phrase_you_know(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I think you know we should do it", FormatMode.CLEAN)
        assert "you know" not in result.formatted_text.lower()
        assert "should do it" in result.formatted_text.lower()

    def test_filler_phrase_so_basically(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("so basically the plan is to launch", FormatMode.CLEAN)
        assert "so basically" not in result.formatted_text.lower()
        assert "plan" in result.formatted_text.lower()

    def test_filler_phrase_i_mean(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I mean it really works great", FormatMode.CLEAN)
        assert "i mean" not in result.formatted_text.lower()
        assert "works great" in result.formatted_text.lower()

    def test_protected_like(self):
        """'like' in 'I like pizza' should NOT be removed."""
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I like pizza and I like code", FormatMode.CLEAN)
        # "like" should be kept — it's a verb meaning "enjoy"
        assert "like pizza" in result.formatted_text.lower() or "like code" in result.formatted_text.lower()

    def test_filler_like_removed(self):
        """'like' in 'could like maybe' IS filler and should be removed."""
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("we could like maybe launch the product on Tuesday", FormatMode.CLEAN)
        assert "like" not in result.formatted_text.lower().split()
        assert "launch" in result.formatted_text.lower()

    def test_filler_like_with_verb_ing(self):
        """'like' before verb-ing is filler: 'was like thinking'"""
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I was like thinking about it", FormatMode.CLEAN)
        assert "like" not in result.formatted_text.lower().split()
        assert "thinking" in result.formatted_text.lower()

    def test_repeated_words(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I went to the the store", FormatMode.CLEAN)
        assert "the the" not in result.formatted_text
        assert result.repeated_words_fixed >= 1

    def test_false_starts(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("I wa- I went to the store", FormatMode.CLEAN)
        assert "wa-" not in result.formatted_text
        assert "went" in result.formatted_text

    def test_auto_capitalization(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("the meeting is tomorrow", FormatMode.CLEAN)
        assert result.formatted_text[0] == "T"

    def test_auto_punctuation(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("the meeting is tomorrow", FormatMode.CLEAN)
        assert result.formatted_text.endswith(".")

    def test_preserves_existing_punctuation(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("Is the meeting tomorrow?", FormatMode.CLEAN)
        assert result.formatted_text.endswith("?")
        assert not result.formatted_text.endswith("?.")

    def test_multiple_fillers_removed(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format(
            "so um basically I was like thinking you know we could sort of do it",
            FormatMode.CLEAN,
        )
        assert result.fillers_removed >= 3
        assert "thinking" in result.formatted_text.lower()
        assert "do it" in result.formatted_text.lower()

    def test_format_result_to_dict(self):
        from axiom_vox.stt.formatter import TextFormatter, FormatMode

        f = TextFormatter()
        result = f.format("um hello", FormatMode.CLEAN)
        d = result.to_dict()
        assert "formatted_text" in d
        assert "original_text" in d
        assert d["mode"] == "clean"
        assert d["fillers_removed"] >= 1

    def test_format_for_ai(self):
        from axiom_vox.stt.formatter import TextFormatter

        f = TextFormatter()
        cleaned, prompt = f.format_for_ai("um so I was thinking about it")
        assert "um" not in cleaned.lower()
        assert "text formatter" in prompt.lower()

    def test_clean_text_convenience(self):
        from axiom_vox.stt.formatter import clean_text

        result = clean_text("um hello world")
        assert "um" not in result.lower()
        assert "hello" in result.lower()


# ============================================================================
# HOTWORD TESTS
# ============================================================================

class TestHotwordManager:
    """Test custom dictionary / hotwords."""

    def test_add_and_list(self):
        from axiom_vox.stt.hotwords import HotwordManager
        hw = HotwordManager(dict_path="/tmp/test_vox_dict.json")
        hw.add("AXIØM", boost=5, corrections=["axium", "axiom", "axiom"])
        assert len(hw.entries) == 1
        assert hw.entries["axiøm"].boost == 5

    def test_remove(self):
        from axiom_vox.stt.hotwords import HotwordManager
        hw = HotwordManager(dict_path="/tmp/test_vox_dict2.json")
        hw.add("VØX", boost=5)
        assert hw.remove("VØX") is True
        assert hw.remove("nonexistent") is False
        assert len(hw.entries) == 0

    def test_hotword_list(self):
        from axiom_vox.stt.hotwords import HotwordManager
        hw = HotwordManager(dict_path="/tmp/test_vox_dict3.json")
        hw.add("AXIØM", boost=5)
        hw.add("VØX", boost=4)
        words = hw.get_hotword_list()
        assert "AXIØM" in words
        assert "VØX" in words

    def test_corrections(self):
        from axiom_vox.stt.hotwords import HotwordManager
        hw = HotwordManager(dict_path="/tmp/test_vox_dict4.json")
        hw.add("AXIØM", corrections=["axium", "axiom"])
        result = hw.apply_corrections("I used axium to build axiom tools")
        assert "AXIØM" in result
        assert "axium" not in result

    def test_no_corrections_when_empty(self):
        from axiom_vox.stt.hotwords import HotwordManager
        hw = HotwordManager(dict_path="/tmp/test_vox_dict_empty.json")
        result = hw.apply_corrections("Hello world")
        assert result == "Hello world"

    def test_entry_to_dict(self):
        from axiom_vox.stt.hotwords import HotwordEntry
        entry = HotwordEntry(word="AXIØM", boost=5, corrections=["axium"])
        d = entry.to_dict()
        assert d["word"] == "AXIØM"
        assert d["boost"] == 5

    def test_persist_and_reload(self):
        import os
        from axiom_vox.stt.hotwords import HotwordManager
        path = "/tmp/test_vox_persist.json"
        if os.path.exists(path):
            os.remove(path)

        hw1 = HotwordManager(dict_path=path)
        hw1.add("TestWord", boost=7)

        hw2 = HotwordManager(dict_path=path)
        assert "testword" in hw2.entries
        assert hw2.entries["testword"].boost == 7


# ============================================================================
# SNIPPET TESTS
# ============================================================================

class TestSnippetManager:
    """Test voice snippets."""

    def test_add_and_expand(self):
        from axiom_vox.stt.snippets import SnippetManager
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip.json")
        sm.snippets.clear()
        sm.add("insert signature", "Best regards,\nJeremy")
        text, matched = sm.expand("please insert signature here")
        assert "Best regards" in text
        assert "insert signature" in matched

    def test_no_match(self):
        from axiom_vox.stt.snippets import SnippetManager
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip2.json")
        sm.snippets.clear()
        sm.add("insert signature", "Best regards")
        text, matched = sm.expand("hello world")
        assert text == "hello world"
        assert matched == []

    def test_remove(self):
        from axiom_vox.stt.snippets import SnippetManager
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip3.json")
        sm.snippets.clear()
        sm.add("test trigger", "expansion")
        assert sm.remove("test trigger") is True
        assert sm.remove("nonexistent") is False

    def test_dynamic_date(self):
        from axiom_vox.stt.snippets import SnippetManager
        from datetime import datetime
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip4.json")
        sm.snippets.clear()
        sm.add("insert date", "{date}")
        text, matched = sm.expand("today is insert date")
        # Should contain current year
        assert str(datetime.now().year) in text
        assert "insert date" in matched

    def test_snippet_to_dict(self):
        from axiom_vox.stt.snippets import Snippet
        s = Snippet(trigger="test", expansion="hello", description="A test")
        d = s.to_dict()
        assert d["trigger"] == "test"
        assert d["expansion"] == "hello"

    def test_disabled_snippet_skipped(self):
        from axiom_vox.stt.snippets import SnippetManager, Snippet
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip5.json")
        sm.snippets.clear()
        sm.snippets["test"] = Snippet(trigger="test", expansion="expanded", enabled=False)
        text, matched = sm.expand("this is a test phrase")
        assert "expanded" not in text
        assert matched == []

    def test_case_insensitive_trigger(self):
        from axiom_vox.stt.snippets import SnippetManager
        sm = SnippetManager(snippets_path="/tmp/test_vox_snip6.json")
        sm.snippets.clear()
        sm.add("Insert Signature", "Best regards")
        text, matched = sm.expand("please INSERT SIGNATURE here")
        assert "Best regards" in text

    def test_import_hotwords(self):
        from axiom_vox.stt import HotwordManager, HotwordEntry, get_hotword_manager
        assert callable(get_hotword_manager)

    def test_import_snippets(self):
        from axiom_vox.stt import SnippetManager, Snippet, get_snippet_manager
        assert callable(get_snippet_manager)


# ============================================================================
# WAKE WORD TESTS
# ============================================================================

class TestWakeWordDetector:
    """Test wake word detection."""

    def test_config_defaults(self):
        from axiom_vox.stt.wakeword import WakeWordConfig
        config = WakeWordConfig()
        assert "vox" in config.wake_words
        assert config.model_size == "tiny"
        assert config.cooldown_seconds == 2.0

    def test_detector_init(self):
        from axiom_vox.stt.wakeword import WakeWordDetector, WakeWordConfig

        triggered = []
        detector = WakeWordDetector(on_wake=lambda: triggered.append(True))
        assert detector.is_listening is False
        assert detector.on_wake is not None

    def test_status_when_not_running(self):
        from axiom_vox.stt.wakeword import WakeWordDetector
        detector = WakeWordDetector(on_wake=lambda: None)
        status = detector.get_status()
        assert status["listening"] is False
        assert "vox" in status["wake_words"]

    def test_cooldown_config(self):
        from axiom_vox.stt.wakeword import WakeWordDetector, WakeWordConfig
        config = WakeWordConfig(cooldown_seconds=5.0)
        detector = WakeWordDetector(on_wake=lambda: None, config=config)
        assert detector.config.cooldown_seconds == 5.0

    def test_custom_wake_words(self):
        from axiom_vox.stt.wakeword import WakeWordConfig
        config = WakeWordConfig(wake_words=["hey vox", "axiom"])
        assert "hey vox" in config.wake_words
        assert "axiom" in config.wake_words
        assert len(config.wake_words) == 2

    def test_import_wakeword(self):
        from axiom_vox.stt import WakeWordDetector, WakeWordConfig, get_wake_detector
        assert callable(get_wake_detector)


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestVoxConfig:
    """Test config persistence."""

    def test_defaults(self):
        from axiom_vox.stt.config import VoxConfig
        cfg = VoxConfig(config_path="/tmp/test_vox_cfg.json")
        assert cfg.get("model") == "base"
        assert cfg.get("format") == "clean"
        assert cfg.get("language") == "auto"

    def test_set_and_get(self):
        from axiom_vox.stt.config import VoxConfig
        cfg = VoxConfig(config_path="/tmp/test_vox_cfg2.json")
        cfg.set("model", "small")
        assert cfg.get("model") == "small"

    def test_bulk_update(self):
        from axiom_vox.stt.config import VoxConfig
        cfg = VoxConfig(config_path="/tmp/test_vox_cfg3.json")
        cfg.update({"model": "large-v3", "format": "professional"})
        assert cfg.get("model") == "large-v3"
        assert cfg.get("format") == "professional"

    def test_reset(self):
        from axiom_vox.stt.config import VoxConfig
        cfg = VoxConfig(config_path="/tmp/test_vox_cfg4.json")
        cfg.set("model", "large-v3")
        cfg.reset()
        assert cfg.get("model") == "base"

    def test_persist_and_reload(self):
        import os
        from axiom_vox.stt.config import VoxConfig
        path = "/tmp/test_vox_cfg_persist.json"
        if os.path.exists(path):
            os.remove(path)

        c1 = VoxConfig(config_path=path)
        c1.set("model", "medium")

        c2 = VoxConfig(config_path=path)
        assert c2.get("model") == "medium"

    def test_to_dict(self):
        from axiom_vox.stt.config import VoxConfig
        cfg = VoxConfig(config_path="/tmp/test_vox_cfg5.json")
        d = cfg.to_dict()
        assert "model" in d
        assert "format" in d
        assert "language" in d

"""
Voice Biometric Embeddings
--------------------------

Speaker embedding extraction for voice biometric verification.

Implements SpectralFingerprint (256-dim, no external dependencies) as baseline,
with optional neural backends (ECAPA-TDNN, Wav2Vec2) for improved accuracy.

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import io
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import numpy as np

try:
    from scipy import signal
    from scipy.fft import fft, rfft
    from scipy.ndimage import uniform_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .models import (
    EmbeddingBackend,
    EmbeddingResult,
    SPECTRAL_EMBEDDING_DIM,
    ECAPA_EMBEDDING_DIM,
    WAV2VEC_EMBEDDING_DIM,
    EMBEDDING_VERSION,
)

logger = logging.getLogger(__name__)


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy embedding to bytes."""
    buffer = io.BytesIO()
    np.save(buffer, embedding.astype(np.float32))
    return buffer.getvalue()


def deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes to numpy embedding."""
    buffer = io.BytesIO(data)
    return np.load(buffer)


class EmbeddingExtractor(ABC):
    """Abstract base class for embedding extractors."""

    @property
    @abstractmethod
    def backend(self) -> EmbeddingBackend:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract embedding from audio."""
        pass

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def extract_and_serialize(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> EmbeddingResult:
        """Extract embedding and return serialized result."""
        try:
            embedding = self.extract(audio, sample_rate)
            duration = len(audio) / sample_rate

            # Estimate quality from audio properties
            rms = np.sqrt(np.mean(audio ** 2))
            quality = min(1.0, rms / 0.1)  # Rough quality estimate

            return EmbeddingResult(
                embedding=serialize_embedding(embedding),
                embedding_dim=self.embedding_dim,
                backend=self.backend,
                quality_score=quality,
                duration_seconds=duration,
                sample_rate=sample_rate,
                is_valid=True,
            )
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return EmbeddingResult(
                embedding=b"",
                embedding_dim=0,
                backend=self.backend,
                is_valid=False,
                message=str(e),
            )


class SpectralFingerprint(EmbeddingExtractor):
    """
    256-dimensional speaker embedding using spectral features.

    No external dependencies beyond numpy/scipy.

    Feature breakdown:
        - MFCCs: 39 dims (13 + 13 deltas + 13 delta-deltas)
        - Spectral stats: 80 dims (40 mel bands × mean/std)
        - Pitch features: 20 dims
        - Formant features: 32 dims (F1-F4 × 8 stats)
        - Temporal features: 25 dims
        - Spectral shape: 60 dims
        Total: 256 dims
    """

    EMBEDDING_DIM = 256
    N_MFCC = 13
    N_MELS = 40
    N_FFT = 2048
    HOP_LENGTH = 512

    def __init__(self):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for SpectralFingerprint")

    @property
    def backend(self) -> EmbeddingBackend:
        return EmbeddingBackend.SPECTRAL

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract 256-dimensional spectral fingerprint."""
        # Ensure float32 and mono
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        features = []

        # 1. MFCCs (39 dims)
        mfccs = self._extract_mfccs(audio, sample_rate)
        features.append(mfccs)

        # 2. Spectral statistics (80 dims)
        spectral_stats = self._extract_spectral_stats(audio, sample_rate)
        features.append(spectral_stats)

        # 3. Pitch features (20 dims)
        pitch_features = self._extract_pitch_features(audio, sample_rate)
        features.append(pitch_features)

        # 4. Formant features (32 dims)
        formant_features = self._extract_formant_features(audio, sample_rate)
        features.append(formant_features)

        # 5. Temporal features (25 dims)
        temporal_features = self._extract_temporal_features(audio, sample_rate)
        features.append(temporal_features)

        # 6. Spectral shape (60 dims)
        spectral_shape = self._extract_spectral_shape(audio, sample_rate)
        features.append(spectral_shape)

        # Concatenate and normalize
        embedding = np.concatenate(features)

        # Pad or truncate to exact dimension
        if len(embedding) < self.EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, self.EMBEDDING_DIM - len(embedding)))
        elif len(embedding) > self.EMBEDDING_DIM:
            embedding = embedding[:self.EMBEDDING_DIM]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _extract_mfccs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features (39 dims)."""
        # Compute mel spectrogram
        mel_spec = self._mel_spectrogram(audio, sr)

        # Log mel spectrogram
        log_mel = np.log(mel_spec + 1e-10)

        # DCT to get MFCCs
        mfccs = self._dct(log_mel, self.N_MFCC)

        # Statistics over time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Delta and delta-delta (approximated with differences)
        delta_mean = np.mean(np.diff(mfccs, axis=1), axis=1) if mfccs.shape[1] > 1 else np.zeros(self.N_MFCC)
        delta_std = np.std(np.diff(mfccs, axis=1), axis=1) if mfccs.shape[1] > 1 else np.zeros(self.N_MFCC)

        # Combine (13 mean + 13 std + 13 delta = 39)
        return np.concatenate([mfcc_mean, mfcc_std, delta_mean])[:39]

    def _extract_spectral_stats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral statistics (80 dims)."""
        mel_spec = self._mel_spectrogram(audio, sr)

        # Mean and std per mel band
        mel_mean = np.mean(mel_spec, axis=1)
        mel_std = np.std(mel_spec, axis=1)

        return np.concatenate([mel_mean, mel_std])[:80]

    def _extract_pitch_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch (F0) features (20 dims)."""
        # Simple autocorrelation-based pitch detection
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        pitches = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            pitch = self._estimate_pitch(frame, sr)
            if pitch > 0:
                pitches.append(pitch)

        if len(pitches) < 2:
            return np.zeros(20)

        pitches = np.array(pitches)

        # Statistics
        features = [
            np.mean(pitches),
            np.std(pitches),
            np.median(pitches),
            np.min(pitches),
            np.max(pitches),
            np.percentile(pitches, 25),
            np.percentile(pitches, 75),
            np.percentile(pitches, 75) - np.percentile(pitches, 25),  # IQR
        ]

        # Pitch contour features
        pitch_diff = np.diff(pitches)
        features.extend([
            np.mean(pitch_diff),
            np.std(pitch_diff),
            np.mean(np.abs(pitch_diff)),
            np.max(np.abs(pitch_diff)),
        ])

        # Pitch histogram (8 bins)
        hist, _ = np.histogram(pitches, bins=8, range=(50, 400))
        hist = hist / (np.sum(hist) + 1e-10)
        features.extend(hist)

        return np.array(features)[:20]

    def _extract_formant_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract formant features (32 dims)."""
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        formants = {i: [] for i in range(4)}  # F1-F4

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            f = self._estimate_formants(frame, sr)
            for j, freq in enumerate(f[:4]):
                if freq > 0:
                    formants[j].append(freq)

        features = []
        for i in range(4):
            if len(formants[i]) > 0:
                f_arr = np.array(formants[i])
                features.extend([
                    np.mean(f_arr),
                    np.std(f_arr),
                    np.median(f_arr),
                    np.min(f_arr),
                    np.max(f_arr),
                    np.percentile(f_arr, 25),
                    np.percentile(f_arr, 75),
                    np.mean(np.abs(np.diff(f_arr))) if len(f_arr) > 1 else 0,
                ])
            else:
                features.extend([0] * 8)

        return np.array(features)[:32]

    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract temporal features (25 dims)."""
        # Energy envelope
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))

        energy = np.array(energy)
        if len(energy) < 2:
            return np.zeros(25)

        # Normalize
        energy = energy / (np.max(energy) + 1e-10)

        features = [
            np.mean(energy),
            np.std(energy),
            np.max(energy),
            np.min(energy),
            np.median(energy),
        ]

        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features.append(zcr)

        # Speech rate approximation (energy peaks)
        threshold = 0.3
        above = energy > threshold
        transitions = np.abs(np.diff(above.astype(int)))
        speech_segments = np.sum(transitions) / 2
        duration = len(audio) / sr
        features.append(speech_segments / duration if duration > 0 else 0)

        # Energy dynamics
        energy_diff = np.diff(energy)
        features.extend([
            np.mean(energy_diff),
            np.std(energy_diff),
            np.mean(np.abs(energy_diff)),
        ])

        # Rhythm features (autocorrelation of energy)
        if len(energy) > 10:
            autocorr = np.correlate(energy - np.mean(energy), energy - np.mean(energy), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            # Find periodicity peaks
            peaks = []
            for i in range(1, min(len(autocorr) - 1, 50)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append((i, autocorr[i]))
            peaks.sort(key=lambda x: x[1], reverse=True)
            for j in range(5):
                if j < len(peaks):
                    features.extend([peaks[j][0], peaks[j][1]])
                else:
                    features.extend([0, 0])
        else:
            features.extend([0] * 10)

        # Silence ratio
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        features.append(silence_ratio)

        return np.array(features)[:25]

    def _extract_spectral_shape(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral shape features (60 dims)."""
        # Compute spectrogram
        n_fft = min(self.N_FFT, len(audio))
        hop = n_fft // 4

        n_frames = max(1, (len(audio) - n_fft) // hop + 1)
        window = signal.windows.hann(n_fft)

        specs = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frame = frame * window
            spec = np.abs(rfft(frame))
            specs.append(spec)

        specs = np.array(specs)
        if len(specs) == 0:
            return np.zeros(60)

        # Average spectrum
        avg_spec = np.mean(specs, axis=0)
        freqs = np.linspace(0, sr / 2, len(avg_spec))

        features = []

        # Spectral centroid
        centroid = np.sum(freqs * avg_spec) / (np.sum(avg_spec) + 1e-10)
        features.append(centroid / sr)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * avg_spec) / (np.sum(avg_spec) + 1e-10))
        features.append(bandwidth / sr)

        # Spectral rolloff (85%, 95%)
        cumsum = np.cumsum(avg_spec)
        total = cumsum[-1] + 1e-10
        rolloff_85 = freqs[np.searchsorted(cumsum, 0.85 * total)]
        rolloff_95 = freqs[np.searchsorted(cumsum, 0.95 * total)]
        features.extend([rolloff_85 / sr, rolloff_95 / sr])

        # Spectral flatness
        log_spec = np.log(avg_spec + 1e-10)
        flatness = np.exp(np.mean(log_spec)) / (np.mean(avg_spec) + 1e-10)
        features.append(flatness)

        # Spectral slope
        if len(freqs) > 1:
            slope = np.polyfit(freqs, avg_spec, 1)[0]
            features.append(slope)
        else:
            features.append(0)

        # Spectral flux (average)
        if len(specs) > 1:
            flux = np.mean(np.sum(np.abs(np.diff(specs, axis=0)), axis=1))
            features.append(flux)
        else:
            features.append(0)

        # Band energies (8 bands)
        band_edges = np.linspace(0, sr / 2, 9)
        for i in range(8):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_energy = np.sum(avg_spec[mask]) / (np.sum(avg_spec) + 1e-10)
            features.append(band_energy)

        # Spectral contrast (8 bands)
        for i in range(8):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_spec = avg_spec[mask]
            if len(band_spec) > 0:
                peak = np.percentile(band_spec, 95)
                valley = np.percentile(band_spec, 5)
                contrast = peak - valley
                features.append(contrast)
            else:
                features.append(0)

        # Spectral variance over time
        spec_var = np.var(specs, axis=0)
        # Sample 32 points
        indices = np.linspace(0, len(spec_var) - 1, 32).astype(int)
        features.extend(spec_var[indices] / (np.max(spec_var) + 1e-10))

        return np.array(features)[:60]

    def _mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel spectrogram."""
        n_fft = min(self.N_FFT, len(audio))
        hop = self.HOP_LENGTH

        # STFT
        n_frames = max(1, (len(audio) - n_fft) // hop + 1)
        window = signal.windows.hann(n_fft)

        specs = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frame = frame * window
            spec = np.abs(rfft(frame)) ** 2
            specs.append(spec)

        power_spec = np.array(specs).T  # (freq, time)

        # Mel filterbank
        mel_fb = self._mel_filterbank(n_fft // 2 + 1, sr)
        mel_spec = np.dot(mel_fb, power_spec)

        return mel_spec

    def _mel_filterbank(self, n_fft_bins: int, sr: int) -> np.ndarray:
        """Create mel filterbank."""
        n_mels = self.N_MELS
        low_freq = 0
        high_freq = sr / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_points = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft_bins * 2 - 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, n_fft_bins))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center != left:
                    fb[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    fb[i, j] = (right - j) / (right - center)

        return fb

    def _dct(self, x: np.ndarray, n_out: int) -> np.ndarray:
        """Compute DCT-II."""
        n = x.shape[0]
        result = np.zeros((n_out, x.shape[1]))

        for k in range(n_out):
            for i in range(n):
                result[k] += x[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
            result[k] *= np.sqrt(2 / n)

        return result

    def _estimate_pitch(self, frame: np.ndarray, sr: int) -> float:
        """Estimate pitch using autocorrelation."""
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        # Find peaks in valid pitch range (80-400 Hz)
        min_lag = int(sr / 400)
        max_lag = int(sr / 80)

        if max_lag >= len(corr):
            return 0.0

        corr_segment = corr[min_lag:max_lag]
        if len(corr_segment) == 0:
            return 0.0

        peak_idx = np.argmax(corr_segment)
        peak_val = corr_segment[peak_idx]

        # Confidence check
        if peak_val < 0.3 * corr[0]:
            return 0.0

        pitch = sr / (min_lag + peak_idx)
        return pitch

    def _estimate_formants(self, frame: np.ndarray, sr: int, n_formants: int = 4) -> List[float]:
        """Estimate formants using LPC."""
        # LPC order
        order = 2 * n_formants + 2

        # Autocorrelation
        n = len(frame)
        r = np.zeros(order + 1)
        for i in range(order + 1):
            r[i] = np.sum(frame[i:] * frame[:n-i])

        # Levinson-Durbin
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            sum_val = sum(a[j] * r[i - j] for j in range(i))
            k = -(r[i] + sum_val) / (e + 1e-10)

            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + k * a[i - j]
            a_new[i] = k

            a = a_new
            e = e * (1 - k * k)

        # Find roots
        roots = np.roots(a)

        # Filter for formants
        formants = []
        for root in roots:
            if np.imag(root) >= 0:
                freq = np.abs(np.arctan2(np.imag(root), np.real(root))) * sr / (2 * np.pi)
                if 200 < freq < sr / 2 - 100:
                    formants.append(freq)

        formants.sort()
        return formants[:n_formants]


class NeuralEmbedding(EmbeddingExtractor):
    """
    Neural speaker embedding using ECAPA-TDNN or Wav2Vec2.

    Requires optional dependencies:
        - ECAPA: speechbrain
        - Wav2Vec2: transformers, torch
    """

    def __init__(self, backend: EmbeddingBackend = EmbeddingBackend.ECAPA):
        self._backend = backend
        self._model = None
        self._load_model()

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    @property
    def embedding_dim(self) -> int:
        if self._backend == EmbeddingBackend.ECAPA:
            return ECAPA_EMBEDDING_DIM
        return WAV2VEC_EMBEDDING_DIM

    def _load_model(self):
        """Load the neural model."""
        if self._backend == EmbeddingBackend.ECAPA:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                logger.info("Loaded ECAPA-TDNN model")
            except ImportError:
                raise ImportError("speechbrain required for ECAPA backend")
        elif self._backend == EmbeddingBackend.WAV2VEC:
            try:
                import torch
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                self._model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self._model.eval()
                logger.info("Loaded Wav2Vec2 model")
            except ImportError:
                raise ImportError("transformers and torch required for Wav2Vec2 backend")

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract neural embedding."""
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (most models expect 16kHz)
        if sample_rate != 16000:
            # Simple resampling
            ratio = 16000 / sample_rate
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio
            )

        if self._backend == EmbeddingBackend.ECAPA:
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            embedding = self._model.encode_batch(audio_tensor)
            return embedding.squeeze().numpy()

        elif self._backend == EmbeddingBackend.WAV2VEC:
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Mean pooling over time
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.squeeze().numpy()

        raise ValueError(f"Unknown backend: {self._backend}")


def get_extractor(backend: EmbeddingBackend = EmbeddingBackend.SPECTRAL) -> EmbeddingExtractor:
    """Get an embedding extractor for the specified backend."""
    if backend == EmbeddingBackend.SPECTRAL:
        return SpectralFingerprint()
    return NeuralEmbedding(backend)


def compute_embedding_hash(embedding: np.ndarray) -> str:
    """Compute hash of embedding for integrity verification."""
    return hashlib.sha256(embedding.tobytes()).hexdigest()[:16]

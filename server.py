import logging
import io
import zipfile
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import noisereduce as nr
import numpy as np
import scipy.signal as signal
import soundfile as sf
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response, jsonify, request, send_file
from flask_compress import Compress
from flask_cors import CORS
from pedalboard import (
    Compressor,
    Delay,
    Gain,
    HighShelfFilter,
    HighpassFilter,
    Limiter,
    LowShelfFilter,
    LowpassFilter,
    PeakFilter,
    Pedalboard,
    Reverb,
)
from PIL import Image
from pydub import AudioSegment
from pypdf import PdfReader
from rembg import remove


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_NAME = "AuraShift AI"
UPLOAD_ROOT = Path(os.getenv("AURASHIFT_TMP_DIR", "/tmp/aurashift"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

DB_PATH = UPLOAD_ROOT / "aurashift_users.db"
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300 MB
TEMP_FILE_TTL_SECONDS = int(os.getenv("TEMP_FILE_TTL_SECONDS", "900"))
CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "10"))
FFMPEG_TIMEOUT_SECONDS = int(os.getenv("FFMPEG_TIMEOUT_SECONDS", "240"))
GHOSTSCRIPT_TIMEOUT_SECONDS = int(os.getenv("GHOSTSCRIPT_TIMEOUT_SECONDS", "240"))
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "")

PAPER_SIZES: Dict[str, Tuple[int, int]] = {
    "A4": (595, 842),
    "A3": (842, 1191),
    "Letter": (612, 792),
    "Legal": (612, 1008),
}


# -----------------------------------------------------------------------------
# App & logging
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

CORS(app)
Compress(app)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(APP_NAME)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------
def create_temp_file(suffix: str) -> Path:
    """Create a unique temporary file path under UPLOAD_ROOT."""
    file_name = f"{uuid.uuid4().hex}{suffix}"
    return UPLOAD_ROOT / file_name


def safe_remove(path: Optional[Path]) -> None:
    """Remove file silently if present."""
    if not path:
        return
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        logger.debug("Failed to cleanup file: %s", path, exc_info=True)


def run_subprocess(command: List[str], timeout: int) -> subprocess.CompletedProcess:
    """Execute a subprocess with timeout and capture output."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def json_error(message: str, status_code: int = 400):
    return jsonify({"ok": False, "error": message}), status_code


def init_whatsapp_db() -> None:
    """Create lightweight user usage table in temporary SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                phone_number TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 0,
                is_premium INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def cleanup_tmp_files() -> None:
    """Periodic cleanup for stale temporary files."""
    now = time.time()
    deleted = 0

    for item in UPLOAD_ROOT.iterdir():
        if not item.is_file():
            continue
        age = now - item.stat().st_mtime
        if age > TEMP_FILE_TTL_SECONDS:
            try:
                item.unlink()
                deleted += 1
            except Exception:
                logger.warning("Could not delete stale temp file: %s", item, exc_info=True)

    logger.info("Temp cleanup completed. Removed files: %s", deleted)


def validate_required_file(field_name: str):
    if field_name not in request.files:
        return None, json_error(f"Missing file field: {field_name}", 400)

    file_obj = request.files[field_name]
    if not file_obj or file_obj.filename == "":
        return None, json_error(f"No file selected for field: {field_name}", 400)

    return file_obj, None


# -----------------------------------------------------------------------------
# Audio pipeline (Ultra Pro Studio — Integrated Vocal Production Chain)
# -----------------------------------------------------------------------------
MAX_STUDIO_DURATION_SECONDS = 300.0  # 5 minutes max for Hugging Face Free Space RAM/CPU safety
MAX_AUDIO_FILE_BYTES = 120 * 1024 * 1024  # 120 MB per audio file


def _parse_float(value, default: float, min_val: float, max_val: float) -> float:
    """Safely parse and clamp a numeric form parameter."""
    try:
        parsed = float(value)
        if not np.isfinite(parsed):
            return default
        return float(np.clip(parsed, min_val, max_val))
    except (TypeError, ValueError):
        return default


def measure_audio_metrics(y_stereo: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Measure sample peak (dBFS), RMS (dBFS), and integrated loudness (LUFS, ITU-R BS.1770 K-weighted).
    Clearly distinguishes sample peak control from measured LUFS loudness.
    """
    if y_stereo.size == 0:
        return {"peak_dbfs": -96.0, "rms_dbfs": -96.0, "lufs": -70.0}

    data = y_stereo if y_stereo.ndim == 2 else np.column_stack((y_stereo, y_stereo))
    peak_lin = float(np.max(np.abs(data)))
    peak_dbfs = 20.0 * np.log10(max(peak_lin, 1e-9))

    rms_lin = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    rms_dbfs = 20.0 * np.log10(max(rms_lin, 1e-9))

    # K-weighting filter stages (High-shelf head model + RLB high-pass)
    try:
        k_board = Pedalboard(
            [
                HighShelfFilter(cutoff_frequency_hz=1500.0, gain_db=4.0, q=0.707),
                HighpassFilter(cutoff_frequency_hz=38.0),
            ]
        )
        k_weighted = k_board(data.astype(np.float32).T, sr).T
    except Exception:
        k_weighted = data

    # 400ms gating blocks (75% overlap -> 100ms step) per BS.1770
    block_size = max(1, int(0.4 * sr))
    step_size = max(1, int(0.1 * sr))
    n_samples = k_weighted.shape[0]

    if n_samples < block_size:
        mean_sq = float(np.mean(np.sum(np.square(k_weighted, dtype=np.float64), axis=1)))
        lufs = -0.691 + 10.0 * np.log10(max(mean_sq, 1e-12))
    else:
        ch_energy = np.sum(np.square(k_weighted, dtype=np.float64), axis=1)
        cumsum = np.concatenate(([0.0], np.cumsum(ch_energy)))
        starts = np.arange(0, n_samples - block_size + 1, step_size)
        block_energies = (cumsum[starts + block_size] - cumsum[starts]) / float(block_size)
        block_lufs = -0.691 + 10.0 * np.log10(np.maximum(block_energies, 1e-12))

        # Absolute gate at -70 LUFS
        abs_mask = block_energies > (10.0 ** ((-70.0 + 0.691) / 10.0))
        if np.any(abs_mask):
            gated_energies = block_energies[abs_mask]
            rel_thresh = -0.691 + 10.0 * np.log10(np.mean(gated_energies)) - 10.0
            rel_mask = block_lufs[abs_mask] > rel_thresh
            final_energies = gated_energies[rel_mask] if np.any(rel_mask) else gated_energies
            lufs = -0.691 + 10.0 * np.log10(max(float(np.mean(final_energies)), 1e-12))
        else:
            lufs = -70.0

    return {
        "peak_dbfs": round(float(np.clip(peak_dbfs, -96.0, 6.0)), 2),
        "rms_dbfs": round(float(np.clip(rms_dbfs, -96.0, 6.0)), 2),
        "lufs": round(float(np.clip(lufs, -70.0, 0.0)), 2),
    }


def apply_ai_autotune(
    y: np.ndarray,
    sr: int,
    strength: float,
    retune_speed: float = 0.5,
    key: str = "C",
    scale: str = "chromatic",
) -> np.ndarray:
    """Conservative phrase-level pitch correction with selectable musical scale."""
    if strength <= 0.01 or len(y) < int(sr * 0.15):
        return y

    key_names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
                 "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
                 "A": 9, "A#": 10, "Bb": 10, "B": 11}
    scale_intervals = {
        "chromatic": list(range(12)),
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "major_pentatonic": [0, 2, 4, 7, 9],
        "minor_pentatonic": [0, 3, 5, 7, 10],
    }
    root = key_names.get(str(key), 0)
    intervals = scale_intervals.get(str(scale), scale_intervals["chromatic"])

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    kernel = 15 if retune_speed >= 0.65 else 21
    f0 = signal.medfilt(f0, kernel_size=kernel)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples", backtrack=True)
    segments = [0] + [int(o) for o in onsets if 0 < int(o) < len(y)] + [len(y)]
    corrected = np.copy(y)
    min_seg_len = int(sr * (0.045 if retune_speed >= 0.65 else 0.065))
    scale_factor = float(np.clip(strength * (0.55 + 0.35 * retune_speed), 0.0, 0.92))
    # Candidate MIDI notes span the singer's normal range and one octave of margin.
    allowed_midi = [octave * 12 + root + interval for octave in range(1, 9) for interval in intervals]

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        segment = y[start:end]
        if len(segment) < min_seg_len:
            continue
        frame_start = max(0, int(librosa.samples_to_frames(np.array([start]), hop_length=512)[0]))
        frame_end = min(len(f0), int(librosa.samples_to_frames(np.array([end]), hop_length=512)[0]))
        seg_f0 = f0[frame_start:frame_end]
        valid_f0 = seg_f0[np.isfinite(seg_f0) & (seg_f0 > 0)]
        if len(valid_f0) < 3:
            continue

        median_f0 = float(np.median(valid_f0))
        if median_f0 <= 0:
            continue
        midi_float = 12.0 * np.log2(median_f0 / 440.0) + 69.0
        target_midi = min(allowed_midi, key=lambda m: abs(m - midi_float))
        shift = float(np.clip((target_midi - midi_float) * scale_factor, -2.5, 2.5))
        if abs(shift) < 0.08:
            continue

        segment_corrected = librosa.effects.pitch_shift(
            segment.astype("float32"), sr=sr, n_steps=shift, res_type="soxr_hq"
        )
        if len(segment_corrected) > len(segment):
            segment_corrected = segment_corrected[:len(segment)]
        elif len(segment_corrected) < len(segment):
            segment_corrected = np.pad(segment_corrected, (0, len(segment) - len(segment_corrected)))

        fade_len = min(int(sr * 0.018), len(segment) // 4)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            segment_corrected[:fade_len] = segment_corrected[:fade_len] * fade_in + segment[:fade_len] * fade_out
            segment_corrected[-fade_len:] = segment_corrected[-fade_len:] * fade_out + segment[-fade_len:] * fade_in
        corrected[start:end] = segment_corrected
    return corrected.astype(np.float32)


def apply_split_band_deesser(y: np.ndarray, sr: int, amount: float, freq_hz: float = 6800.0) -> np.ndarray:
    """
    Dynamic split-band sibilance reduction (4.8kHz - 10.5kHz).
    Uses zero-phase bandpass isolation (sosfiltfilt) so sibilance subtraction is phase-exact,
    dynamically attenuating harsh 'S'/'T' peaks without dulling non-sibilant vocal passages.
    """
    if amount <= 0.01 or len(y) < 64:
        return y

    nyq = 0.5 * sr
    low_hz = float(np.clip(freq_hz * 0.78, 3800.0, nyq * 0.8))
    high_hz = float(np.clip(freq_hz * 1.45, low_hz + 1200.0, nyq * 0.94))

    sos_bp = signal.butter(2, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    if y.ndim == 1:
        sib_band = signal.sosfiltfilt(sos_bp, y).astype(np.float32)
        det = np.abs(sib_band)
    else:
        sib_band = np.column_stack(
            [signal.sosfiltfilt(sos_bp, y[:, ch]) for ch in range(y.shape[1])]
        ).astype(np.float32)
        det = np.max(np.abs(sib_band), axis=1)

    # Smooth envelope (~6ms attack/release window)
    win_len = max(3, int(sr * 0.006))
    kernel = np.ones(win_len, dtype=np.float32) / float(win_len)
    env = np.convolve(det, kernel, mode="same")

    rms_ref = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))) + 1e-7)
    thresh = rms_ref * (0.85 - 0.45 * amount)
    thresh = max(thresh, 0.005)

    # Compute gain reduction only when sibilance exceeds threshold
    over = np.maximum(0.0, env - thresh) / (env + 1e-7)
    max_reduction = float(np.clip(0.75 * amount, 0.0, 0.75))  # Up to ~12 dB sibilance taming
    reduction = np.clip(over * (0.55 + 0.65 * amount), 0.0, max_reduction).astype(np.float32)

    if y.ndim == 1:
        return (y - sib_band * reduction).astype(np.float32)
    return (y - sib_band * reduction[:, np.newaxis]).astype(np.float32)


def apply_harmonic_exciter(y: np.ndarray, sr: int, exciter_amount: float, warmth_amount: float) -> np.ndarray:
    """
    Dual-stage harmonic saturation:
    - Vocal Warmth: subtle low-mid tube/tape soft-clipping saturation (120Hz - 1.2kHz)
    - Harmonic Exciter: high-frequency air/shimmer harmonic generation (>2.6kHz)
    """
    out = y.astype(np.float32)
    nyq = 0.5 * sr

    # 1. Optional Vocal Warmth (tube/tape low-mid harmonic body)
    if warmth_amount > 0.01:
        drive_w = 1.0 + warmth_amount * 2.2
        norm_w = float(np.tanh(drive_w))
        sos_lp = signal.butter(2, min(1400.0 / nyq, 0.9), btype="lowpass", output="sos")
        if out.ndim == 1:
            low_mid = signal.sosfilt(sos_lp, out)
        else:
            low_mid = np.column_stack([signal.sosfilt(sos_lp, out[:, ch]) for ch in range(out.shape[1])])
        warm_sat = (np.tanh(low_mid * drive_w) / norm_w) - low_mid
        out = out + (warm_sat * (warmth_amount * 0.38)).astype(np.float32)

    # 2. Harmonic Exciter (upper-mid & air harmonic shimmer)
    if exciter_amount > 0.01:
        drive_e = 1.2 + exciter_amount * 3.0
        norm_e = float(np.tanh(drive_e))
        sos_hp = signal.butter(2, min(2600.0 / nyq, 0.9), btype="highpass", output="sos")
        if out.ndim == 1:
            high_band = signal.sosfilt(sos_hp, out)
        else:
            high_band = np.column_stack([signal.sosfilt(sos_hp, out[:, ch]) for ch in range(out.shape[1])])
        # Asymmetric + odd harmonic excitation for upper presence
        excited = (np.tanh(high_band * drive_e) / norm_e) + 0.15 * np.square(high_band) * np.sign(high_band)
        if out.ndim == 1:
            excited = signal.sosfilt(sos_hp, excited)
        else:
            excited = np.column_stack([signal.sosfilt(sos_hp, excited[:, ch]) for ch in range(out.shape[1])])
        out = out + (excited * (exciter_amount * 0.28)).astype(np.float32)

    return out


def apply_stereo_doubler(y_stereo: np.ndarray, sr: int, width_amount: float) -> np.ndarray:
    """
    Phase-safe Haas + micro-pitch modulated stereo vocal doubler.
    Preserves mono center vocal intelligibility while adding lush left/right stereo width.
    """
    if width_amount <= 0.01:
        return y_stereo

    mono = 0.5 * (y_stereo[:, 0] + y_stereo[:, 1])
    n_samples = len(mono)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)

    # Left voice: ~14ms base delay + slow 0.45Hz pitch/time modulation
    # Right voice: ~22ms base delay + 0.62Hz opposite-phase modulation
    delay_l_samples = (0.0135 + 0.0012 * np.sin(2.0 * np.pi * 0.45 * t)) * sr
    delay_r_samples = (0.0215 + 0.0014 * np.cos(2.0 * np.pi * 0.62 * t)) * sr

    idx = np.arange(n_samples, dtype=np.float32)
    idx_l = np.clip(idx - delay_l_samples, 0.0, n_samples - 1.0)
    idx_r = np.clip(idx - delay_r_samples, 0.0, n_samples - 1.0)

    # Linear interpolation for smooth fractional sample delay
    i0_l = np.floor(idx_l).astype(np.int32)
    frac_l = idx_l - i0_l
    i1_l = np.minimum(i0_l + 1, n_samples - 1)
    left_dbl = (1.0 - frac_l) * mono[i0_l] + frac_l * mono[i1_l]

    i0_r = np.floor(idx_r).astype(np.int32)
    frac_r = idx_r - i0_r
    i1_r = np.minimum(i0_r + 1, n_samples - 1)
    right_dbl = (1.0 - frac_r) * mono[i0_r] + frac_r * mono[i1_r]

    # Band-limit the doubled voices (180Hz - 7.5kHz) so they sit behind the lead vocal
    dbl_pair = np.column_stack((left_dbl, right_dbl)).astype(np.float32)
    dbl_filter = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=180.0),
            LowpassFilter(cutoff_frequency_hz=7500.0),
        ]
    )
    dbl_pair = dbl_filter(dbl_pair.T, sr).T

    blend = float(np.clip(width_amount * 0.32, 0.0, 0.34))
    out_l = y_stereo[:, 0] + dbl_pair[:, 0] * blend
    out_r = y_stereo[:, 1] + dbl_pair[:, 1] * blend
    return np.column_stack((out_l, out_r)).astype(np.float32)


def apply_tempo_vocal_delay(
    y_stereo: np.ndarray,
    sr: int,
    delay_mix: float,
    bpm: float = 120.0,
    subdivision: str = "1/4",
    feedback: float = 0.28,
) -> np.ndarray:
    """
    Tempo-adjustable stereo vocal delay with filtered repeats (160Hz - 4.2kHz)
    so echoes sit cleanly under the lead vocal without muddying lyrics.
    Preserves exact input sample count.
    """
    if delay_mix <= 0.01:
        return y_stereo

    bpm_clamped = float(np.clip(bpm, 50.0, 220.0))
    quarter_sec = 60.0 / bpm_clamped
    subdiv_map = {
        "1/16": 0.25,
        "1/8": 0.5,
        "1/8d": 0.75,
        "1/4": 1.0,
        "1/4t": 2.0 / 3.0,
        "1/2": 2.0,
        "slap": 0.095 / quarter_sec,
    }
    mult = subdiv_map.get(subdivision, 1.0)
    delay_sec = float(np.clip(quarter_sec * mult, 0.045, 1.25))
    fb_clamped = float(np.clip(feedback, 0.0, 0.68))

    # Subtle stereo offset on right channel for spatial depth
    delay_l = Delay(delay_seconds=delay_sec, feedback=fb_clamped, mix=1.0)
    delay_r = Delay(delay_seconds=min(1.35, delay_sec * 1.03), feedback=fb_clamped * 0.94, mix=1.0)

    filter_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=180.0),
            LowpassFilter(cutoff_frequency_hz=4200.0),
        ]
    )
    filtered_in = filter_board(y_stereo.T, sr).T
    wet_l = delay_l(filtered_in[:, 0:1].T, sr).T[:, 0]
    wet_r = delay_r(filtered_in[:, 1:2].T, sr).T[:, 0]
    wet_stereo = np.column_stack((wet_l, wet_r)).astype(np.float32)

    # Keep wet delay musical and tucked beneath lead vocal
    wet_gain = float(np.clip((delay_mix / 100.0) * 0.36, 0.0, 0.38))
    return (y_stereo + wet_stereo * wet_gain).astype(np.float32)


def mix_vocal_with_instrumental(
    vocal_stereo: np.ndarray,
    inst_stereo: Optional[np.ndarray],
    sr: int,
    vocal_gain_db: float = 0.0,
    inst_gain_db: float = -3.0,
    carve_eq_amount: float = 0.35,
    backing_vocal_audio: Optional[np.ndarray] = None,
    backing_level_db: float = -9.0,
    backing_pan: float = 0.0,
    sidechain_amount: float = 0.0,
    instrumental_offset_ms: float = 0.0,
    backing_offset_ms: float = 0.0,
    return_stems: bool = False,
):
    """Build a sample-aligned vocal/backing/instrumental mix; optionally return pre-master stems."""
    def stereo(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if audio is None or np.size(audio) == 0:
            return None
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.column_stack((arr, arr))
        elif arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
        return arr

    def offset_track(arr: Optional[np.ndarray], offset_ms: float) -> Optional[np.ndarray]:
        if arr is None:
            return None
        offset_samples = int(round(float(np.clip(offset_ms, -3000.0, 3000.0)) * sr / 1000.0))
        if offset_samples > 0:
            return np.pad(arr, ((offset_samples, 0), (0, 0)), mode="constant").astype(np.float32)
        if offset_samples < 0:
            trim = min(-offset_samples, len(arr))
            return arr[trim:, :].astype(np.float32, copy=True)
        return arr

    vocal = stereo(vocal_stereo)
    inst = offset_track(stereo(inst_stereo), instrumental_offset_ms)
    backing = offset_track(stereo(backing_vocal_audio), backing_offset_ms)
    if vocal is None:
        vocal = np.zeros((0, 2), dtype=np.float32)

    lengths = [len(vocal)] + ([len(inst)] if inst is not None else []) + ([len(backing)] if backing is not None else [])
    target_len = max(lengths) if lengths else 0

    def fit_length(arr: Optional[np.ndarray]) -> np.ndarray:
        if arr is None:
            return np.zeros((target_len, 2), dtype=np.float32)
        if len(arr) >= target_len:
            return arr[:target_len].astype(np.float32, copy=True)
        return np.pad(arr, ((0, target_len - len(arr)), (0, 0)), mode="constant").astype(np.float32)

    vocal_gain = float(10.0 ** (np.clip(vocal_gain_db, -18.0, 12.0) / 20.0))
    vocal_stem = fit_length(vocal) * vocal_gain

    backing_stem = np.zeros((target_len, 2), dtype=np.float32)
    if backing is not None:
        back = fit_length(backing)
        back_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=120.0),
            PeakFilter(cutoff_frequency_hz=280.0, gain_db=-1.5, q=1.0),
            Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=15.0, release_ms=120.0),
        ])
        back = back_board(back.T, sr).T.astype(np.float32)
        back_mono = np.mean(back, axis=1)
        pan = float(np.clip(backing_pan, -100.0, 100.0))
        angle = (pan + 100.0) / 200.0 * (np.pi / 2.0)
        pan_pair = np.column_stack((np.full(target_len, np.cos(angle)), np.full(target_len, np.sin(angle)))).astype(np.float32)
        back_gain = float(10.0 ** (np.clip(backing_level_db, -30.0, 6.0) / 20.0))
        backing_stem = back_mono[:, None] * pan_pair * back_gain

    instrumental_stem = np.zeros((target_len, 2), dtype=np.float32)
    if inst is not None:
        inst_aligned = fit_length(inst)
        carve_db = float(np.clip(-2.2 * carve_eq_amount, -3.5, 0.0))
        inst_board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=2400.0, gain_db=carve_db, q=0.9),
            Gain(gain_db=float(np.clip(inst_gain_db, -24.0, 6.0))),
        ])
        instrumental_stem = inst_board(inst_aligned.T, sr).T.astype(np.float32)

        # Optional vocal-triggered ducking: subtle reduction only while the lead is active.
        sc = float(np.clip(sidechain_amount / 100.0, 0.0, 1.0))
        if sc > 0.001 and target_len:
            lead_env = np.max(np.abs(vocal_stem), axis=1).astype(np.float32)
            cutoff = min(35.0 / (0.5 * sr), 0.95)
            sos = signal.butter(2, cutoff, btype="lowpass", output="sos")
            env = signal.sosfilt(sos, lead_env)
            ref = float(np.percentile(env, 90)) if env.size else 0.0
            if ref > 1e-5:
                env_norm = np.clip(env / (ref * 0.65), 0.0, 1.0)
                max_duck_db = 3.5 * sc
                duck_gain = np.power(10.0, (-max_duck_db * env_norm) / 20.0).astype(np.float32)
                instrumental_stem *= duck_gain[:, None]

    mix = (vocal_stem + backing_stem + instrumental_stem).astype(np.float32)
    if return_stems:
        stems = {"vocal": vocal_stem, "backing_vocal": backing_stem, "instrumental": instrumental_stem}
        return mix, stems
    return mix


def apply_studio_fx(
    y: np.ndarray,
    sr: int,
    noise_red: float = 0.0,
    autotune_strength: float = 0.15,
    reverb_amount: float = 14.0,
    bass_amount: float = 50.0,
    vocal_intensity: float = 55.0,
    vocal_sharpness: float = 54.0,
    *,
    retune_speed: float = 50.0,
    pitch_shift_semitones: float = 0.0,
    pitch_key: str = "C",
    pitch_scale: str = "chromatic",
    time_stretch_rate: float = 1.0,
    low_cut_hz: float = 85.0,
    mud_cut_db: float = -2.0,
    presence_db_override: Optional[float] = None,
    air_db_override: Optional[float] = None,
    comp_amount: float = 45.0,
    manual_comp_enabled: bool = False,
    comp_threshold_db: float = -18.0,
    comp_ratio_override: float = 3.0,
    comp_attack_ms: float = 15.0,
    comp_release_ms: float = 100.0,
    deesser_amount: float = 35.0,
    exciter_amount: float = 20.0,
    warmth_amount: float = 25.0,
    doubler_amount: float = 12.0,
    delay_mix: float = 10.0,
    delay_bpm: float = 120.0,
    delay_subdivision: str = "1/4",
    delay_feedback: float = 25.0,
    reverb_decay: float = 38.0,
    vocal_level_db: float = 0.0,
    instrumental_audio: Optional[np.ndarray] = None,
    inst_level_db: float = -3.5,
    backing_vocal_audio: Optional[np.ndarray] = None,
    backing_level_db: float = -9.0,
    backing_pan: float = 0.0,
    sidechain_amount: float = 0.0,
    instrumental_offset_ms: float = 0.0,
    backing_offset_ms: float = 0.0,
    bus_glue_amount: float = 35.0,
    ceiling_dbfs: float = -1.0,
    return_stems: bool = False,
):
    """
    Integrated Professional Vocal-Production Chain for Ultra Pro Studio:
    1. Input gain staging & DC offset removal (no noise reduction gate artifacts on clean vocals)
    2. Natural Pitch Correction (scale-chromatic median pitch tracking with smooth crossfades)
    3. Smart Vocal EQ (High-pass rumble filter, low-mid mud cleanup, body warmth, presence, air)
    4. Studio Vocal Compression (2-stage leveling + transient control)
    5. Split-Band De-Esser (dynamic sibilance control around 6.8kHz)
    6. Harmonic Exciter & Optional Tube/Tape Vocal Warmth
    7. Stereo Vocal Doubler (phase-safe Haas + micro-modulation widening)
    8. Tempo-Adjustable Vocal Delay (filtered stereo repeats)
    9. Studio Reverb (high-passed lush plate/room reflection with controlled wet cap)
    10. Vocal Level & Optional Instrumental/Beat Mixing (with vocal pocket EQ carving)
    11. Final Bus Glue Compression & True-Ceiling Peak Limiter (preserving output headroom)
    """
    if y.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Keep a mono lead-vocal processing path; source files remain untouched.
    y_mono = np.mean(y, axis=1).astype(np.float32) if y.ndim > 1 else y.astype(np.float32)
    y_mono = y_mono - float(np.mean(y_mono))

    # Optional denoise is deliberately opt-in: too much reduction can create watery artifacts.
    denoise_amount = float(np.clip(noise_red / 100.0, 0.0, 1.0))
    if denoise_amount > 0.005:
        try:
            y_mono = nr.reduce_noise(y=y_mono, sr=sr, prop_decrease=denoise_amount, stationary=False).astype(np.float32)
        except Exception:
            logger.warning("Studio noise reduction failed; continuing with the source signal.", exc_info=True)

    # Optional tempo/time stretch. Rate > 1 speeds up/shortens; rate < 1 slows down/lengthens.
    stretch_rate = float(np.clip(time_stretch_rate, 0.85, 1.15))
    if abs(stretch_rate - 1.0) > 0.002:
        y_mono = librosa.effects.time_stretch(y_mono.astype(np.float32), rate=stretch_rate).astype(np.float32)

    # Global pitch transposition is separate from note correction.
    pitch_shift = float(np.clip(pitch_shift_semitones, -12.0, 12.0))
    if abs(pitch_shift) >= 0.05:
        y_mono = librosa.effects.pitch_shift(y_mono.astype(np.float32), sr=sr, n_steps=pitch_shift, res_type="soxr_hq").astype(np.float32)

    # Input gain staging after cleanup/stretch/transpose provides processing headroom.
    max_val = float(np.max(np.abs(y_mono))) if y_mono.size else 0.0
    if max_val > 1e-6:
        y_mono = (y_mono / max_val) * 0.707

    # Musical pitch correction snaps to the selected key/scale only when enabled.
    pitch_strength = float(np.clip(autotune_strength, 0.0, 1.0))
    speed_norm = float(np.clip(retune_speed / 100.0, 0.0, 1.0))
    if pitch_strength > 0.01:
        y_mono = apply_ai_autotune(y_mono, sr, pitch_strength, retune_speed=speed_norm, key=pitch_key, scale=pitch_scale)
    n_original = len(y_mono)

    # 3. Smart Vocal EQ & Studio Vocal Compression
    # Map legacy & extended parameters musically so conservative defaults preserve natural tone
    bass_db = float(np.clip(((bass_amount - 50.0) / 50.0) * 4.5, -5.0, 5.0))
    intensity_norm = float(np.clip((vocal_intensity - 50.0) / 50.0, -1.0, 1.0))
    sharpness_norm = float(np.clip((vocal_sharpness - 50.0) / 50.0, -1.0, 1.0))

    presence_gain = (
        float(np.clip(presence_db_override, -6.0, 6.0))
        if presence_db_override is not None
        else float(np.clip(intensity_norm * 4.5, -4.5, 5.0))
    )
    crisp_gain = float(np.clip(sharpness_norm * 3.5, -4.5, 4.5))
    air_gain = (
        float(np.clip(air_db_override, -6.0, 6.0))
        if air_db_override is not None
        else float(np.clip(sharpness_norm * 4.0, -4.5, 5.0))
    )
    mud_db = float(np.clip(mud_cut_db, -6.0, 2.0))
    hp_hz = float(np.clip(low_cut_hz, 40.0, 160.0))

    # Studio Compressor parameters (avoid over-compression)
    comp_norm = float(np.clip(comp_amount / 100.0, 0.0, 1.0))
    comp_ratio = float(np.clip(1.6 + comp_norm * 2.8 + max(0.0, intensity_norm) * 0.8, 1.2, 5.0))
    comp_thresh = float(np.clip(-14.0 - comp_norm * 10.0, -26.0, -10.0))
    comp_attack = 12.0
    comp_release = 110.0
    if manual_comp_enabled:
        comp_ratio = float(np.clip(comp_ratio_override, 1.2, 8.0))
        comp_thresh = float(np.clip(comp_threshold_db, -36.0, -6.0))
        comp_attack = float(np.clip(comp_attack_ms, 1.0, 80.0))
        comp_release = float(np.clip(comp_release_ms, 20.0, 400.0))
    makeup_db = float(np.clip(comp_norm * 3.2 + intensity_norm * 1.8, -2.0, 5.5))

    eq_comp_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=hp_hz),
            LowShelfFilter(cutoff_frequency_hz=135.0, gain_db=bass_db, q=0.707),
            PeakFilter(cutoff_frequency_hz=280.0, gain_db=mud_db, q=1.2),
            PeakFilter(cutoff_frequency_hz=2850.0, gain_db=presence_gain, q=0.95),
            PeakFilter(cutoff_frequency_hz=5200.0, gain_db=crisp_gain, q=1.1),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=air_gain, q=0.707),
            Compressor(
                threshold_db=comp_thresh,
                ratio=comp_ratio,
                attack_ms=comp_attack,
                release_ms=comp_release,
            ),
            Gain(gain_db=makeup_db),
        ]
    )
    y_vocal = eq_comp_board(y_mono, sr)

    # 4. Split-Band De-Esser (post-EQ/compressor sibilance control)
    deess_norm = float(np.clip(deesser_amount / 100.0, 0.0, 1.0))
    if deess_norm > 0.01:
        y_vocal = apply_split_band_deesser(y_vocal, sr, amount=deess_norm, freq_hz=6800.0)

    # 5. Harmonic Exciter & Optional Vocal Warmth
    exciter_norm = float(np.clip(exciter_amount / 100.0, 0.0, 1.0))
    warmth_norm = float(np.clip(warmth_amount / 100.0, 0.0, 1.0))
    if exciter_norm > 0.01 or warmth_norm > 0.01:
        y_vocal = apply_harmonic_exciter(y_vocal, sr, exciter_amount=exciter_norm, warmth_amount=warmth_norm)

    # Expand to stereo for spatial processing
    y_stereo = np.column_stack((y_vocal, y_vocal)).astype(np.float32)

    # 6. Stereo Doubler
    doubler_norm = float(np.clip(doubler_amount / 100.0, 0.0, 1.0))
    if doubler_norm > 0.01:
        y_stereo = apply_stereo_doubler(y_stereo, sr, width_amount=doubler_norm)

    # 7. Tempo-Adjustable Vocal Delay
    if delay_mix > 0.5:
        y_stereo = apply_tempo_vocal_delay(
            y_stereo,
            sr,
            delay_mix=float(np.clip(delay_mix, 0.0, 100.0)),
            bpm=delay_bpm,
            subdivision=delay_subdivision,
            feedback=float(np.clip(delay_feedback / 100.0, 0.0, 0.68)),
        )

    # 8. Studio Reverb (conservative wet cap to prevent washed-out vocals)
    rev_norm = float(np.clip(reverb_amount / 100.0, 0.0, 1.0))
    if rev_norm > 0.01:
        room_sz = float(np.clip(0.22 + (reverb_decay / 100.0) * 0.52, 0.18, 0.78))
        wet_mix = float(np.clip(rev_norm * 0.28, 0.0, 0.30))
        wet_board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=320.0),
                LowpassFilter(cutoff_frequency_hz=7800.0),
                Reverb(room_size=room_sz, damping=0.72, wet_level=1.0, dry_level=0.0, width=0.92),
            ]
        )
        y_wet = wet_board(y_stereo.T, sr).T
        y_stereo = (y_stereo * (1.0 - wet_mix * 0.25) + y_wet * wet_mix).astype(np.float32)

    # 9. Track mixing: lead vocal, optional backing layer and optional instrumental.
    y_mix, pre_master_stems = mix_vocal_with_instrumental(
        y_stereo,
        instrumental_audio,
        sr,
        vocal_gain_db=vocal_level_db,
        inst_gain_db=inst_level_db,
        carve_eq_amount=0.35,
        backing_vocal_audio=backing_vocal_audio,
        backing_level_db=backing_level_db,
        backing_pan=backing_pan,
        sidechain_amount=sidechain_amount,
        instrumental_offset_ms=instrumental_offset_ms,
        backing_offset_ms=backing_offset_ms,
        return_stems=True,
    )

    # 10. Final Bus Glue Compression & Sample-Ceiling Peak Limiter
    glue_norm = float(np.clip(bus_glue_amount / 100.0, 0.0, 1.0))
    ceiling_clamped = float(np.clip(ceiling_dbfs, -6.0, -0.2))

    master_plugins = []
    if glue_norm > 0.02:
        glue_ratio = float(np.clip(1.2 + glue_norm * 1.4, 1.15, 2.6))
        glue_thresh = float(np.clip(-10.0 - glue_norm * 8.0, -19.0, -8.0))
        master_plugins.append(
            Compressor(
                threshold_db=glue_thresh,
                ratio=glue_ratio,
                attack_ms=25.0,
                release_ms=140.0,
            )
        )
        master_plugins.append(Gain(gain_db=float(glue_norm * 1.2)))

    master_plugins.append(Limiter(threshold_db=ceiling_clamped, release_ms=75.0))
    master_board = Pedalboard(master_plugins)
    y_out = master_board(y_mix.T, sr).T

    # Final safety sample ceiling. This is not a substitute for a dedicated true-peak meter.
    hard_ceiling_lin = float(10.0 ** (ceiling_clamped / 20.0))
    peak_now = float(np.max(np.abs(y_out))) if y_out.size > 0 else 0.0
    if peak_now > hard_ceiling_lin and peak_now > 0:
        y_out = (y_out / peak_now) * hard_ceiling_lin

    y_out = y_out.astype(np.float32)
    if return_stems:
        # Preserve stem balance by applying one common safety attenuation if any premaster stem exceeds 0 dBFS.
        stem_peak = max((float(np.max(np.abs(stem))) for stem in pre_master_stems.values() if stem.size), default=0.0)
        if stem_peak > 0.89:
            stem_scale = 0.89 / stem_peak
            pre_master_stems = {name: (stem * stem_scale).astype(np.float32) for name, stem in pre_master_stems.items()}
        return {"mix": y_out, **pre_master_stems}
    return y_out


# -----------------------------------------------------------------------------
# Request lifecycle hooks
# -----------------------------------------------------------------------------
@app.before_request
def log_request() -> None:
    logger.info("Request: %s %s", request.method, request.path)


@app.errorhandler(413)
def payload_too_large(_error):
    return json_error("Uploaded file is too large. Limit is 300MB.", 413)


@app.errorhandler(404)
def not_found(_error):
    return json_error("Route not found.", 404)


@app.errorhandler(500)
def internal_error(_error):
    return json_error("Internal server error.", 500)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    return send_file("indexx.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": APP_NAME})


@app.get("/api/tools")
def list_tools():
    tools = [
        "image_resizer_client",
        "pdf_compressor",
        "qr_creator_client",
        "image_to_pdf_client",
        "pdf_merger_client",
        "pdf_to_text_doc",
        "background_remover",
        "ultra_pro_studio_manual_vocal",
        "ultra_pro_studio_clean",
        "ultra_pro_studio_retro_tape_vinyl",
        "ultra_pro_studio_ambient_delay",
        "ultra_pro_studio_analog_synth",
        "photo_watermark_remover",
        "video_watermark_remover",
    ]
    return jsonify({"ok": True, "tools": tools})


@app.route("/webhook", methods=["GET", "POST"], strict_slashes=False)
def webhook_handler():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode and token and mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
            return Response(challenge, mimetype="text/plain"), 200

        if not WEBHOOK_VERIFY_TOKEN:
            return "Webhook token is not configured.", 503

        return "Forbidden", 403

    raw_data = request.get_data(as_text=True)
    logger.info("Webhook event length: %s", len(raw_data))
    return "EVENT_RECEIVED", 200


@app.post("/compress_pdf")
def compress_pdf_route():
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        pdf_file, err = validate_required_file("pdf_file")
        if err:
            return err

        target_kb = request.form.get("target_kb", "100")
        try:
            target_kb_int = max(10, min(100000, int(target_kb)))
        except ValueError:
            target_kb_int = 100

        input_path = create_temp_file("_input.pdf")
        output_path = create_temp_file("_compressed.pdf")
        pdf_file.save(input_path)

        quality = "/screen" if target_kb_int <= 300 else "/ebook"

        gs_path = subprocess.run(
            ["which", "gs"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if not gs_path:
            return json_error(
                "Ghostscript not available. Install ghostscript in your runtime image.",
                500,
            )

        cmd = [
            gs_path,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={quality}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={output_path}",
            str(input_path),
        ]

        proc = run_subprocess(cmd, timeout=GHOSTSCRIPT_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            logger.error("Ghostscript error: %s", proc.stderr)
            return json_error("PDF compression failed in Ghostscript.", 500)

        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"aurashift_compressed_{target_kb_int}kb.pdf",
            mimetype="application/pdf",
        )
    except subprocess.TimeoutExpired:
        return json_error("PDF compression timed out.", 504)
    except Exception:
        logger.error("compress_pdf_route failed", exc_info=True)
        return json_error("Unexpected error during PDF compression.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


@app.post("/image_to_pdf")
def image_to_pdf_route():
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        image_file, err = validate_required_file("image_file")
        if err:
            return err

        paper_size_name = request.form.get("paper_size", "A4")
        page_dimensions = PAPER_SIZES.get(paper_size_name, PAPER_SIZES["A4"])

        input_path = create_temp_file("_input_image")
        output_path = create_temp_file("_output.pdf")
        image_file.save(input_path)

        image = Image.open(input_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.thumbnail(page_dimensions, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", page_dimensions, (255, 255, 255))
        x_offset = (page_dimensions[0] - image.width) // 2
        y_offset = (page_dimensions[1] - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        canvas.save(output_path, "PDF", resolution=100.0)

        return send_file(
            output_path,
            as_attachment=True,
            download_name="aurashift_image_to_pdf.pdf",
            mimetype="application/pdf",
        )
    except Exception:
        logger.error("image_to_pdf_route failed", exc_info=True)
        return json_error("Image to PDF conversion failed.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


@app.post("/pdf_to_doc")
def pdf_to_doc():
    """Extract text from PDF and return TXT or DOCX."""
    import docx

    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        pdf_file, err = validate_required_file("pdf_file")
        if err:
            return err

        output_format = request.form.get("format", "txt").lower()
        if output_format not in {"txt", "docx"}:
            return json_error("Unsupported output format. Use txt or docx.", 400)

        input_path = create_temp_file("_extract.pdf")
        pdf_file.save(input_path)

        reader = PdfReader(str(input_path))
        extracted_pages: List[str] = []
        for page in reader.pages:
            extracted_pages.append(page.extract_text() or "")

        content = "\n\n".join(extracted_pages).strip()
        if not content:
            return json_error(
                "No extractable text found. PDF might be scanned/image-only.",
                422,
            )

        if output_format == "txt":
            output_path = create_temp_file("_extracted.txt")
            output_path.write_text(content, encoding="utf-8")
            return send_file(
                output_path,
                as_attachment=True,
                download_name="aurashift_extracted.txt",
                mimetype="text/plain",
            )

        output_path = create_temp_file("_extracted.docx")
        document = docx.Document()
        document.add_heading("AuraShift Extracted Text", level=1)
        for paragraph in content.split("\n"):
            document.add_paragraph(paragraph)
        document.save(str(output_path))

        return send_file(
            output_path,
            as_attachment=True,
            download_name="aurashift_extracted.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception:
        logger.error("pdf_to_doc failed", exc_info=True)
        return json_error("PDF text extraction failed.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


@app.post("/studio/analyze")
def analyze_studio_audio():
    """Estimate tempo for the Studio delay clock; estimate is advisory, not ground truth."""
    temp_files: List[Path] = []
    try:
        audio_file, err = validate_required_file("audio_file")
        if err:
            return err
        raw_path = create_temp_file("_studio_analysis_raw")
        wav_path = create_temp_file("_studio_analysis.wav")
        temp_files.extend([raw_path, wav_path])
        audio_file.save(raw_path)
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            return json_error("Uploaded audio file is empty.", 400)
        if raw_path.stat().st_size > MAX_AUDIO_FILE_BYTES:
            return json_error("Audio file exceeds 120 MB limit.", 413)
        try:
            AudioSegment.from_file(raw_path).export(wav_path, format="wav")
            y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=90.0)
        except Exception:
            return json_error("Could not decode the audio file for tempo analysis.", 400)
        if y.size < int(sr * 2.0):
            return json_error("Please use an audio clip at least 2 seconds long for tempo analysis.", 400)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_candidates = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
        bpm = float(np.atleast_1d(tempo_candidates)[0])
        bpm = float(np.clip(bpm, 50.0, 220.0))
        return jsonify({"ok": True, "estimated_bpm": round(bpm, 1), "analyzed_seconds": round(len(y) / sr, 1), "note": "Tempo estimates may be half-time or double-time; verify by listening to the instrumental."})
    except Exception:
        logger.error("Studio tempo analysis failed", exc_info=True)
        return json_error("Studio tempo analysis failed.", 500)
    finally:
        for path in temp_files:
            safe_remove(path)


@app.post("/autotune")
def autotune_audio():
    """
    Process vocal audio using the integrated Ultra Pro Studio vocal-production chain
    with optional sample-aligned instrumental mixing, real-time peak/LUFS metering,
    and configurable MP3/WAV export.
    """
    temp_files: List[Path] = []

    try:
        action = request.form.get("action", "vocal_chain").strip().lower()
        if action not in {"clean", "manual", "vocal_chain"}:
            return json_error("Invalid action. Use vocal_chain, manual, or clean.", 400)

        audio_file, err = validate_required_file("audio_file")
        if err:
            return err

        raw_input_path = create_temp_file("_raw_input")
        input_path = create_temp_file("_input.wav")
        temp_files.extend([raw_input_path, input_path])

        audio_file.save(raw_input_path)
        if not raw_input_path.exists() or raw_input_path.stat().st_size == 0:
            return json_error("Uploaded vocal audio file is empty.", 400)
        if raw_input_path.stat().st_size > MAX_AUDIO_FILE_BYTES:
            return json_error("Vocal audio file exceeds 120 MB limit.", 413)

        try:
            AudioSegment.from_file(raw_input_path).export(input_path, format="wav")
            y, sr = librosa.load(input_path, sr=44100, mono=True)
        except Exception:
            return json_error(
                "Invalid or unsupported vocal audio file. Please upload WAV, MP3, M4A, FLAC, or OGG.",
                400,
            )

        if y.size == 0 or len(y) < int(sr * 0.1):
            return json_error("Vocal recording is too short or silent to process.", 400)

        duration_sec = float(len(y)) / float(sr)
        if duration_sec > MAX_STUDIO_DURATION_SECONDS:
            return json_error(
                f"Vocal track is {duration_sec:.1f}s. Maximum duration is {int(MAX_STUDIO_DURATION_SECONDS)}s (5 minutes) for Free Space memory stability.",
                400,
            )

        in_metrics = measure_audio_metrics(y, sr)

        # Optional instrumental/beat upload
        inst_audio: Optional[np.ndarray] = None
        inst_file = request.files.get("instrumental_file")
        if inst_file and inst_file.filename and inst_file.filename.strip() != "":
            raw_inst_path = create_temp_file("_raw_inst")
            inst_wav_path = create_temp_file("_inst.wav")
            temp_files.extend([raw_inst_path, inst_wav_path])
            inst_file.save(raw_inst_path)
            if raw_inst_path.exists() and raw_inst_path.stat().st_size > 0:
                if raw_inst_path.stat().st_size > MAX_AUDIO_FILE_BYTES:
                    return json_error("Instrumental audio file exceeds 120 MB limit.", 413)
                try:
                    AudioSegment.from_file(raw_inst_path).export(inst_wav_path, format="wav")
                    y_inst_loaded, _ = librosa.load(inst_wav_path, sr=sr, mono=False)
                    if y_inst_loaded.size > 0:
                        if y_inst_loaded.ndim == 1:
                            inst_audio = np.column_stack((y_inst_loaded, y_inst_loaded)).astype(np.float32)
                        else:
                            inst_audio = y_inst_loaded[:2, :].T.astype(np.float32)
                except Exception:
                    return json_error(
                        "Invalid or unsupported instrumental audio file. Please upload a valid audio file.",
                        400,
                    )

        # Optional backing-vocal/double track; kept separate from the lead for stem export.
        backing_audio: Optional[np.ndarray] = None
        backing_file = request.files.get("backing_vocal_file")
        if backing_file and backing_file.filename and backing_file.filename.strip() != "":
            raw_back_path = create_temp_file("_raw_backing")
            backing_wav_path = create_temp_file("_backing.wav")
            temp_files.extend([raw_back_path, backing_wav_path])
            backing_file.save(raw_back_path)
            if raw_back_path.exists() and raw_back_path.stat().st_size > 0:
                if raw_back_path.stat().st_size > MAX_AUDIO_FILE_BYTES:
                    return json_error("Backing-vocal file exceeds 120 MB limit.", 413)
                try:
                    AudioSegment.from_file(raw_back_path).export(backing_wav_path, format="wav")
                    y_back_loaded, _ = librosa.load(backing_wav_path, sr=sr, mono=False)
                    if y_back_loaded.size > 0:
                        if y_back_loaded.ndim == 1:
                            backing_audio = np.column_stack((y_back_loaded, y_back_loaded)).astype(np.float32)
                        else:
                            backing_audio = y_back_loaded[:2, :].T.astype(np.float32)
                except Exception:
                    return json_error("Invalid or unsupported backing-vocal file. Please upload a valid audio file.", 400)

        # Parse all vocal-production chain parameters with conservative, musical defaults
        autotune = _parse_float(request.form.get("autotune"), 18.0, 0.0, 100.0) / 100.0
        retune_speed = _parse_float(request.form.get("retune_speed"), 48.0, 0.0, 100.0)
        pitch_shift_semitones = _parse_float(request.form.get("pitch_shift_semitones"), 0.0, -12.0, 12.0)
        pitch_key = request.form.get("pitch_key", "C").strip()
        if pitch_key not in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}:
            pitch_key = "C"
        pitch_scale = request.form.get("pitch_scale", "chromatic").strip().lower()
        if pitch_scale not in {"chromatic", "major", "minor", "major_pentatonic", "minor_pentatonic"}:
            pitch_scale = "chromatic"
        time_stretch_rate = _parse_float(request.form.get("time_stretch_rate"), 100.0, 85.0, 115.0) / 100.0
        if duration_sec / time_stretch_rate > MAX_STUDIO_DURATION_SECONDS:
            return json_error("The selected time-stretch would make the rendered vocal exceed the 5-minute processing limit. Choose a smaller stretch change or use a shorter file.", 400)
        noise_reduction = _parse_float(request.form.get("noise_reduction"), 0.0, 0.0, 100.0)
        low_cut_hz = _parse_float(request.form.get("low_cut_hz"), 85.0, 40.0, 160.0)
        bass = _parse_float(request.form.get("bass"), 52.0, 0.0, 100.0)
        mud_cut_db = _parse_float(request.form.get("mud_cut_db"), -2.0, -6.0, 2.0)
        vocal_intensity = _parse_float(request.form.get("vocal_intensity"), 56.0, 0.0, 100.0)
        vocal_sharpness = _parse_float(request.form.get("vocal_sharpness"), 54.0, 0.0, 100.0)

        presence_raw = request.form.get("presence_db")
        presence_db_override = (
            _parse_float(presence_raw, 1.5, -6.0, 6.0) if presence_raw is not None and presence_raw != "" else None
        )
        air_raw = request.form.get("air_db")
        air_db_override = (
            _parse_float(air_raw, 1.2, -6.0, 6.0) if air_raw is not None and air_raw != "" else None
        )

        comp_amount = _parse_float(request.form.get("comp_amount"), 45.0, 0.0, 100.0)
        manual_comp_enabled = request.form.get("manual_comp_enabled", "false").strip().lower() in {"1", "true", "yes", "on"}
        comp_threshold_db = _parse_float(request.form.get("comp_threshold_db"), -18.0, -36.0, -6.0)
        comp_ratio_override = _parse_float(request.form.get("comp_ratio"), 3.0, 1.2, 8.0)
        comp_attack_ms = _parse_float(request.form.get("comp_attack_ms"), 15.0, 1.0, 80.0)
        comp_release_ms = _parse_float(request.form.get("comp_release_ms"), 100.0, 20.0, 400.0)
        deesser_amount = _parse_float(request.form.get("deesser_amount"), 35.0, 0.0, 100.0)
        exciter_amount = _parse_float(request.form.get("exciter_amount"), 20.0, 0.0, 100.0)
        warmth_amount = _parse_float(request.form.get("warmth_amount"), 25.0, 0.0, 100.0)
        doubler_amount = _parse_float(request.form.get("doubler_amount"), 12.0, 0.0, 100.0)
        delay_mix = _parse_float(request.form.get("delay_mix"), 10.0, 0.0, 100.0)
        delay_bpm = _parse_float(request.form.get("delay_bpm"), 120.0, 50.0, 220.0)
        delay_subdivision = request.form.get("delay_subdivision", "1/4").strip()
        if delay_subdivision not in {"1/16", "1/8", "1/8d", "1/4", "1/4t", "1/2", "slap"}:
            delay_subdivision = "1/4"
        delay_feedback = _parse_float(request.form.get("delay_feedback"), 25.0, 0.0, 75.0)
        reverb = _parse_float(request.form.get("reverb"), 14.0, 0.0, 100.0)
        reverb_decay = _parse_float(request.form.get("reverb_decay"), 38.0, 10.0, 90.0)
        vocal_level_db = _parse_float(request.form.get("vocal_level_db"), 0.0, -12.0, 8.0)
        inst_level_db = _parse_float(request.form.get("inst_level_db"), -3.5, -24.0, 6.0)
        backing_level_db = _parse_float(request.form.get("backing_level_db"), -9.0, -30.0, 6.0)
        backing_pan = _parse_float(request.form.get("backing_pan"), 0.0, -100.0, 100.0)
        sidechain_amount = _parse_float(request.form.get("sidechain_amount"), 0.0, 0.0, 100.0)
        instrumental_offset_ms = _parse_float(request.form.get("instrumental_offset_ms"), 0.0, -3000.0, 3000.0)
        backing_offset_ms = _parse_float(request.form.get("backing_offset_ms"), 0.0, -3000.0, 3000.0)
        bus_glue_amount = _parse_float(request.form.get("bus_glue_amount"), 35.0, 0.0, 100.0)
        ceiling_dbfs = _parse_float(request.form.get("ceiling_dbfs"), -1.0, -3.0, -0.2)

        export_format = request.form.get("export_format", "mp3").strip().lower()
        if export_format not in {"mp3", "wav", "wav24", "stems_zip"}:
            export_format = "mp3"

        rendered = apply_studio_fx(
            y,
            sr,
            noise_red=noise_reduction,
            autotune_strength=autotune,
            reverb_amount=reverb,
            bass_amount=bass,
            vocal_intensity=vocal_intensity,
            vocal_sharpness=vocal_sharpness,
            retune_speed=retune_speed,
            pitch_shift_semitones=pitch_shift_semitones,
            pitch_key=pitch_key,
            pitch_scale=pitch_scale,
            time_stretch_rate=time_stretch_rate,
            low_cut_hz=low_cut_hz,
            mud_cut_db=mud_cut_db,
            presence_db_override=presence_db_override,
            air_db_override=air_db_override,
            comp_amount=comp_amount,
            manual_comp_enabled=manual_comp_enabled,
            comp_threshold_db=comp_threshold_db,
            comp_ratio_override=comp_ratio_override,
            comp_attack_ms=comp_attack_ms,
            comp_release_ms=comp_release_ms,
            deesser_amount=deesser_amount,
            exciter_amount=exciter_amount,
            warmth_amount=warmth_amount,
            doubler_amount=doubler_amount,
            delay_mix=delay_mix,
            delay_bpm=delay_bpm,
            delay_subdivision=delay_subdivision,
            delay_feedback=delay_feedback,
            reverb_decay=reverb_decay,
            vocal_level_db=vocal_level_db,
            instrumental_audio=inst_audio,
            inst_level_db=inst_level_db,
            backing_vocal_audio=backing_audio,
            backing_level_db=backing_level_db,
            backing_pan=backing_pan,
            sidechain_amount=sidechain_amount,
            instrumental_offset_ms=instrumental_offset_ms,
            backing_offset_ms=backing_offset_ms,
            bus_glue_amount=bus_glue_amount,
            ceiling_dbfs=ceiling_dbfs,
            return_stems=(export_format == "stems_zip"),
        )

        if isinstance(rendered, dict):
            y_processed = rendered["mix"]
            stem_arrays = {name: arr for name, arr in rendered.items() if name != "mix"}
        else:
            y_processed = rendered
            stem_arrays = {}
        duration_sec = float(len(y_processed)) / float(sr)
        out_metrics = measure_audio_metrics(y_processed, sr)

        clean_wav = create_temp_file("_clean.wav")
        temp_files.append(clean_wav)
        wav_subtype = "PCM_24" if export_format in {"wav24", "stems_zip"} else "PCM_16"
        sf.write(clean_wav, np.clip(y_processed, -1.0, 1.0).astype(np.float32), sr, subtype=wav_subtype)

        if export_format == "stems_zip":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(clean_wav, arcname="AuraShift_Studio_Master_24bit.wav")
                stem_titles = {
                    "vocal": "Lead_Vocal_Premaster_24bit.wav",
                    "backing_vocal": "Backing_Vocal_Premaster_24bit.wav",
                    "instrumental": "Instrumental_Premaster_24bit.wav",
                }
                for stem_name, stem_audio in stem_arrays.items():
                    if stem_audio is None or not stem_audio.size or float(np.max(np.abs(stem_audio))) < 1e-8:
                        continue
                    stem_buffer = io.BytesIO()
                    sf.write(stem_buffer, np.clip(stem_audio, -1.0, 1.0).astype(np.float32), sr, format="WAV", subtype="PCM_24")
                    stem_buffer.seek(0)
                    archive.writestr(stem_titles.get(stem_name, f"{stem_name}_24bit.wav"), stem_buffer.read())
            zip_buffer.seek(0)
            resp = send_file(zip_buffer, as_attachment=True, download_name="AuraShift_Studio_Stems.zip", mimetype="application/zip")
            out_name = "AuraShift_Studio_Stems.zip"
        elif export_format in {"wav", "wav24"}:
            send_path = clean_wav
            mime_type = "audio/wav"
            out_name = "AuraShift_UltraPro_VocalMaster_24bit.wav" if export_format == "wav24" else "AuraShift_UltraPro_VocalMaster_16bit.wav"
            resp = send_file(send_path, as_attachment=True, download_name=out_name, mimetype=mime_type)
        else:
            clean_mp3 = create_temp_file("_clean.mp3")
            temp_files.append(clean_mp3)
            AudioSegment.from_wav(clean_wav).export(clean_mp3, format="mp3", bitrate="320k")
            send_path = clean_mp3
            mime_type = "audio/mpeg"
            if action == "manual":
                out_name = "AuraShift_ManualVocal.mp3"
            elif action == "clean":
                out_name = "AuraShift_StudioClean.mp3"
            else:
                out_name = "AuraShift_UltraPro_VocalMaster.mp3"
            resp = send_file(send_path, as_attachment=True, download_name=out_name, mimetype=mime_type)
        # Attach peak vs integrated LUFS loudness telemetry headers for frontend live readout
        resp.headers["X-Audio-Duration-Sec"] = f"{duration_sec:.2f}"
        resp.headers["X-Audio-Sample-Rate"] = str(sr)
        resp.headers["X-Audio-Input-Peak-DBFS"] = f"{in_metrics['peak_dbfs']:.2f}"
        resp.headers["X-Audio-Input-LUFS"] = f"{in_metrics['lufs']:.2f}"
        resp.headers["X-Audio-Output-Peak-DBFS"] = f"{out_metrics['peak_dbfs']:.2f}"
        resp.headers["X-Audio-Output-RMS-DBFS"] = f"{out_metrics['rms_dbfs']:.2f}"
        resp.headers["X-Audio-Output-LUFS"] = f"{out_metrics['lufs']:.2f}"
        resp.headers["X-Audio-Ceiling-DBFS"] = f"{ceiling_dbfs:.2f}"
        resp.headers["X-Audio-Has-Instrumental"] = "true" if inst_audio is not None else "false"
        resp.headers["X-Audio-Has-Backing-Vocal"] = "true" if backing_audio is not None else "false"
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Audio-Duration-Sec, X-Audio-Sample-Rate, X-Audio-Input-Peak-DBFS, "
            "X-Audio-Input-LUFS, X-Audio-Output-Peak-DBFS, X-Audio-Output-RMS-DBFS, "
            "X-Audio-Output-LUFS, X-Audio-Ceiling-DBFS, X-Audio-Has-Instrumental, X-Audio-Has-Backing-Vocal"
        )
        return resp
    except MemoryError:
        logger.error("autotune_audio ran out of memory", exc_info=True)
        return json_error("Audio track is too large for available memory. Try a shorter clip.", 413)
    except Exception:
        logger.error("autotune_audio failed", exc_info=True)
        return json_error("Ultra Pro Studio processing failed.", 500)
    finally:
        for path in temp_files:
            safe_remove(path)


@app.post("/remove_bg")
def remove_bg_route():
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        image_file, err = validate_required_file("image")
        if err:
            return err

        input_path = create_temp_file("_input_image")
        output_path = create_temp_file("_no_bg.png")
        image_file.save(input_path)

        with input_path.open("rb") as source:
            output_data = remove(source.read())

        output_path.write_bytes(output_data)

        return send_file(
            output_path,
            as_attachment=True,
            download_name="aurashift_nobg.png",
            mimetype="image/png",
        )
    except Exception:
        logger.error("remove_bg_route failed", exc_info=True)
        return json_error("Background removal failed.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


@app.post("/remove_photo_watermark")
def remove_photo_watermark_route():
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        image_file, err = validate_required_file("image")
        if err:
            return err

        x = int(float(request.form.get("x", 0)))
        y = int(float(request.form.get("y", 0)))
        w = int(float(request.form.get("w", 50)))
        h = int(float(request.form.get("h", 50)))
        pad = int(request.form.get("padding", 5))
        blur = int(request.form.get("blur", 0))

        input_path = create_temp_file("_wm_photo_input.png")
        output_path = create_temp_file("_wm_photo_output.png")
        image_file.save(input_path)

        image = cv2.imread(str(input_path))
        if image is None:
            return json_error("Failed to read uploaded image.", 400)

        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        if pad > 0:
            kernel = np.ones((pad, pad), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

        if blur > 0:
            blur_kernel = blur if blur % 2 != 0 else blur + 1
            roi = result[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)

        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return send_file(
            output_path,
            as_attachment=True,
            download_name="aurashift_clean_photo.png",
            mimetype="image/png",
        )
    except Exception:
        logger.error("remove_photo_watermark_route failed", exc_info=True)
        return json_error("Photo watermark removal failed.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


@app.post("/remove_video_watermark")
def remove_video_watermark_route():
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    try:
        video_file, err = validate_required_file("video")
        if err:
            return err

        input_path = create_temp_file("_wm_video_input.mp4")
        output_path = create_temp_file("_wm_video_output.mp4")
        video_file.save(input_path)

        cap = cv2.VideoCapture(str(input_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_w <= 0 or frame_h <= 0:
            return json_error("Invalid video file.", 400)

        pad = int(request.form.get("padding", 5))
        blur = int(request.form.get("blur", 0))

        px = max(0, int(float(request.form.get("x", 0))) - pad)
        py = max(0, int(float(request.form.get("y", 0))) - pad)
        req_w = int(float(request.form.get("w", 50))) + (pad * 2)
        req_h = int(float(request.form.get("h", 50))) + (pad * 2)

        pw = min(req_w, frame_w - px)
        ph = min(req_h, frame_h - py)

        if pw <= 0 or ph <= 0:
            return json_error("Watermark area is outside video bounds.", 400)

        if blur > 0:
            filter_str = (
                f"[0:v]crop={pw}:{ph}:{px}:{py},"
                f"boxblur=luma_radius={blur}:chroma_radius={blur}[crop];"
                f"[0:v][crop]overlay={px}:{py}"
            )
        else:
            filter_str = f"delogo=x={px}:y={py}:w={pw}:h={ph}:show=0"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_str,
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            str(output_path),
        ]

        proc = run_subprocess(cmd, timeout=FFMPEG_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            logger.error("FFmpeg error: %s", proc.stderr)
            return json_error("Video processing failed.", 500)

        return send_file(
            output_path,
            as_attachment=True,
            download_name="aurashift_clean_video.mp4",
            mimetype="video/mp4",
        )
    except subprocess.TimeoutExpired:
        return json_error("Video processing timed out.", 504)
    except Exception:
        logger.error("remove_video_watermark_route failed", exc_info=True)
        return json_error("Video watermark removal failed.", 500)
    finally:
        safe_remove(input_path)
        safe_remove(output_path)


# -----------------------------------------------------------------------------
# Startup tasks
# -----------------------------------------------------------------------------
init_whatsapp_db()
cleanup_scheduler = BackgroundScheduler(daemon=True)
cleanup_scheduler.add_job(cleanup_tmp_files, trigger="interval", minutes=CLEANUP_INTERVAL_MINUTES)
cleanup_scheduler.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)

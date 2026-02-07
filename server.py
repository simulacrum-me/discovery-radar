import os
import subprocess
import logging
from mcp.server.fastmcp import FastMCP

import numpy as np
import torch

# Fix for PyTorch 2.6+ weights_only=True default
# Allow omegaconf types used by whisperx/pyannote model checkpoints
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import ValueNode
torch.serialization.add_safe_globals([
    DictConfig, ListConfig, OmegaConf,
    ContainerMetadata, Metadata, ValueNode
])

import whisperx
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

mcp = FastMCP("transcriber")

# Module-level caches for loaded models
_whisper_model = None
_whisper_model_size = None
_embedding_model = None

SAMPLE_RATE = 16000  # whisperx uses 16kHz audio


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_compute_type():
    return "float16" if torch.cuda.is_available() else "int8"


def _load_whisper_model(model_size: str):
    global _whisper_model, _whisper_model_size
    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model
    logger.info("Loading WhisperX model: %s", model_size)
    _whisper_model = whisperx.load_model(
        model_size, _get_device(), compute_type=_get_compute_type()
    )
    _whisper_model_size = model_size
    return _whisper_model


def _load_embedding_model():
    """Load SpeechBrain ECAPA-TDNN speaker embedding model (local, no token needed)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    logger.info("Loading speaker embedding model (ECAPA-TDNN)...")
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    _embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": _get_device()},
    )
    return _embedding_model


def _extract_segment_embedding(embedding_model, audio: np.ndarray, start: float, end: float):
    """Extract speaker embedding for an audio segment."""
    start_sample = int(start * SAMPLE_RATE)
    end_sample = int(end * SAMPLE_RATE)
    segment_audio = audio[start_sample:end_sample]

    # Skip segments shorter than 0.5s — too short for reliable embedding
    if len(segment_audio) < int(SAMPLE_RATE * 0.5):
        return None

    waveform = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        embedding = embedding_model.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()


def _diarize_segments(audio: np.ndarray, segments: list, min_speakers: int | None, max_speakers: int | None) -> list:
    """Assign speaker labels to segments using speechbrain embeddings + agglomerative clustering."""
    embedding_model = _load_embedding_model()

    embeddings = []
    valid_indices = []

    for i, seg in enumerate(segments):
        emb = _extract_segment_embedding(embedding_model, audio, seg["start"], seg["end"])
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    if len(embeddings) == 0:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    if len(embeddings) == 1:
        segments[valid_indices[0]]["speaker"] = "SPEAKER_00"
        for seg in segments:
            seg.setdefault("speaker", "SPEAKER_00")
        return segments

    emb_matrix = normalize(np.array(embeddings))

    # Determine clustering strategy
    if max_speakers is not None:
        n_clusters = min(max_speakers, len(emb_matrix))
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    elif min_speakers is not None:
        n_clusters = min(min_speakers, len(emb_matrix))
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        # Auto-detect number of speakers using distance threshold
        # On L2-normalized embeddings, threshold ~1.0 ≈ cosine similarity 0.5
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0,
        )

    labels = clustering.fit_predict(emb_matrix)

    # Map cluster IDs to sequential speaker labels
    label_to_speaker = {}
    for idx, valid_idx in enumerate(valid_indices):
        label = labels[idx]
        if label not in label_to_speaker:
            label_to_speaker[label] = f"SPEAKER_{len(label_to_speaker):02d}"
        segments[valid_idx]["speaker"] = label_to_speaker[label]

    # Assign nearest speaker to any skipped (too-short) segments
    for seg in segments:
        if "speaker" not in seg:
            seg["speaker"] = "SPEAKER_00"

    return segments


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


@mcp.tool()
def transcribe_video(
    file_path: str,
    model_size: str = "base",
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> str:
    """Transcribe an MP4 video file with speaker diarization.

    Returns a speaker-labeled transcript with timestamps.
    All models run locally — no API keys or tokens required.

    Args:
        file_path: Absolute path to the MP4 file.
        model_size: Whisper model size (tiny, base, small, medium, large-v2).
        language: Language code (e.g. 'en', 'ru') or None for auto-detect.
        min_speakers: Minimum expected number of speakers.
        max_speakers: Maximum expected number of speakers.
    """
    # 1. Validate file
    if not os.path.isfile(file_path):
        return f"Error: file not found: {file_path}"

    # 2. Load whisper model
    model = _load_whisper_model(model_size)

    # 3. Load audio
    logger.info("Loading audio: %s", file_path)
    audio = whisperx.load_audio(file_path)

    # 4. Transcribe
    logger.info("Transcribing...")
    transcribe_kwargs = {}
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, batch_size=16, **transcribe_kwargs)

    detected_language = result.get("language", language or "en")

    # 5. Align
    logger.info("Aligning timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language, device=_get_device()
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        _get_device(),
        return_char_alignments=False,
    )

    # 6. Diarize (local speechbrain embeddings + clustering)
    logger.info("Running speaker diarization...")
    segments = _diarize_segments(audio, result["segments"], min_speakers, max_speakers)

    # 7. Format output
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{speaker}] ({start} - {end}): {text}")

    transcript = "\n".join(lines)
    logger.info("Transcription complete: %d segments", len(lines))
    return transcript


@mcp.tool()
def download_video(
    url: str,
    output_dir: str | None = None,
    filename: str | None = None,
) -> str:
    """Download a video from YouTube (or other supported sites) as MP4 using yt-dlp.

    Returns the absolute path to the downloaded file.

    Args:
        url: YouTube URL or any yt-dlp supported URL.
        output_dir: Directory to save the file. Defaults to current working directory.
        filename: Output filename without extension. Defaults to the video title.
    """
    output_dir = output_dir or os.getcwd()
    if not os.path.isdir(output_dir):
        return f"Error: output directory not found: {output_dir}"

    if filename:
        output_template = os.path.join(output_dir, f"{filename}.%(ext)s")
    else:
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--print", "after_move:filepath",
        "--no-simulate",
        url,
    ]

    logger.info("Downloading video: %s", url)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError:
        return "Error: yt-dlp is not installed. Install it with: pip install yt-dlp"
    except subprocess.TimeoutExpired:
        return "Error: download timed out after 10 minutes"

    if result.returncode != 0:
        return f"Error: yt-dlp failed:\n{result.stderr.strip()}"

    file_path = result.stdout.strip().splitlines()[-1]
    logger.info("Downloaded: %s", file_path)
    return file_path


if __name__ == "__main__":
    mcp.run(transport="stdio")

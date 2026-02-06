# MP4 Transcription MCP Server

MCP server that transcribes MP4 video files with speaker diarization. All models run locally — no API keys or tokens required.

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- CUDA-capable GPU (optional, falls back to CPU)

## Installation

```bash
pip install -r requirements.txt
```

On first run, the following models will be automatically downloaded and cached locally:
- Whisper model (selected size) — for transcription
- SpeechBrain ECAPA-TDNN — for speaker embeddings (cached in `pretrained_models/`)

## Verify Installation

```bash
python -c "import server"
```

## Claude Code MCP Configuration

Add to your Claude Code MCP settings (`.claude/settings.json` or project settings):

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "C:\\Users\\kamaevd\\Desktop\\PO_INVESTIGATIONS\\CLAUDE_CODE"
    }
  }
}
```

## Usage

Once configured, use the `transcribe_video` tool in Claude Code:

```
Transcribe the video at C:\path\to\video.mp4
```

### Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | (required) | Absolute path to MP4 file |
| `model_size` | string | `"base"` | Whisper model: tiny, base, small, medium, large-v2 |
| `language` | string | auto-detect | Language code (e.g. `en`, `ru`) |
| `min_speakers` | int | None | Minimum expected speakers |
| `max_speakers` | int | None | Maximum expected speakers |

### Output Format

```
[SPEAKER_00] (0:00.50 - 0:03.20): Hello, welcome to the meeting.
[SPEAKER_01] (0:03.80 - 0:06.10): Thanks for having me.
```

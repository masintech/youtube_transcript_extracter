# YouTube Transcript Extractor & Summarizer

Two scripts for pulling YouTube transcripts and generating AI summaries:

- **`YoutubeTranscriptionExtrator.py`** — CLI tool: extract and save transcripts from the command line.
- **`YoutubeTranscriptSummarizer.py`** — Gradio web app: extract transcripts and generate AI-powered summaries.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

| Key | Required for |
|-----|-------------|
| `GOOGLE_API_KEY` | Video metadata (title, channel, view count) |
| `OPENAI_API_KEY` | OpenAI model in the Gradio app |
| `ANTHROPIC_API_KEY` | Claude model in the Gradio app |
| `DEEPSEEK_API_KEY` | DeepSeek model in the Gradio app |

The CLI extractor only needs `GOOGLE_API_KEY` (when saving with metadata).

---

## CLI — `YoutubeTranscriptionExtrator.py`

### Usage

```
python YoutubeTranscriptionExtrator.py <url> [options]
```

| Flag | Description |
|------|-------------|
| `-l`, `--save` | Save transcript to a Markdown file (default: print to stdout) |
| `--out FILE` | Output file path (implies `-l`; default: `<video title>.md`) |
| `--lang LANG [LANG ...]` | Preferred transcript language(s) in priority order |
| `-c`, `--copy` | Copy transcript to clipboard instead of printing |

### Examples

```bash
# Print transcript to stdout
python YoutubeTranscriptionExtrator.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Save to a Markdown file (auto-named from video title)
python YoutubeTranscriptionExtrator.py https://youtu.be/dQw4w9WgXcQ -l

# Save to a specific file
python YoutubeTranscriptionExtrator.py https://youtu.be/dQw4w9WgXcQ --out notes.md

# Prefer French transcript, fall back to English
python YoutubeTranscriptionExtrator.py https://youtu.be/dQw4w9WgXcQ --lang fr en

# Copy transcript directly to clipboard
python YoutubeTranscriptionExtrator.py https://youtu.be/dQw4w9WgXcQ -c
```

Supported URL formats: `youtube.com/watch?v=ID`, `youtu.be/ID`, `youtube.com/shorts/ID`.

---

## Gradio App — `YoutubeTranscriptSummarizer.py`

### Launch

```bash
python YoutubeTranscriptSummarizer.py
```

Opens a local web UI at `http://127.0.0.1:7860`.

### How to use

1. Paste a YouTube URL into the input box.
2. Select a model from the dropdown.
3. Click **Generate Transcript and Summary**.
4. The transcript and AI-generated summary stream in side by side.
5. Download the resulting `.md` files via the download buttons.

### Available models

| Model | Provider | Notes |
|-------|----------|-------|
| DeepSeek | DeepSeek API | Default |
| Claude | Anthropic | `claude-sonnet-4-6` |
| OpenAI | OpenAI | `gpt-4o` |
| Ollama | Local | Requires `ollama serve` running locally |

---

## Example workflow

```bash
# Quick clipboard copy to paste into another tool
python YoutubeTranscriptionExtrator.py https://youtu.be/dQw4w9WgXcQ -c

# Save a French transcript for a non-English video
python YoutubeTranscriptionExtrator.py https://youtu.be/VIDEO_ID --lang fr --out transcript_fr.md

# Launch the web app for AI summarization
python YoutubeTranscriptSummarizer.py
```

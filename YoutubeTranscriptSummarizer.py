import gradio as gr
import os
import re
import openai
import anthropic
from datetime import date
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from YoutubeTranscriptionExtrator import (
    extract_video_id,
    get_video_metadata,
    get_youtube_transcript,
    save_transcript_as_markdown,
    _safe_filename,
)

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY', 'your-key-if-not-using-env')

API_TIMEOUT = 120


def _format_timestamped_transcript(entries, video_id):
    lines = []
    for entry in entries:
        start = int(entry['start'])
        m, s = divmod(start, 60)
        url = f"https://youtu.be/{video_id}?t={start}"
        lines.append(f"[{m}:{s:02d}]({url}) {entry['text']}")
    return "\n".join(lines)


def _normalize_mode(mode):
    """Coerce legacy boolean lecture_mode values to the new string format."""
    if mode is True:
        return "Lecture"
    if not mode or mode not in ("General", "Lecture", "Technical"):
        return "General"
    return mode


def _depth_instruction(transcript_text):
    """Return a length-scaled depth rule to prevent over-compression."""
    word_count = len(transcript_text.split())
    if word_count > 8000:
        return (
            "This is a long, dense transcript. "
            "Do NOT compress or skim — expand every concept fully. "
            "Cover every distinct topic the speaker addresses; each Key Concept section must be at least 3–5 sentences.\n\n"
        )
    elif word_count > 3000:
        return (
            "This transcript is moderately long. "
            "Provide complete explanations — do not over-summarise. "
            "Each concept deserves at least 2–3 sentences.\n\n"
        )
    return ""


def _strip_frontmatter(text):
    """Remove YAML frontmatter block for display (keep original for saving)."""
    if text.startswith('---'):
        end = text.find('\n---\n', 3)
        if end != -1:
            return text[end + 5:].lstrip('\n')
    return text


LANGUAGE_CONFIG = {
    "English": {
        "codes": ["en", "en-US", "en-GB", "en-CA", "en-AU"],
        "instruction": "Write your summary in English.",
    },
    "Chinese": {
        "codes": ["zh-Hant", "zh"],
        "instruction": "Write your summary in Traditional Chinese (繁體中文).",
    },
}

DEFAULT_LANGUAGE_CHOICES = ["English", "Chinese"]


def _resolve_lang_cfg(language_choice, detected_langs=None):
    """Return lang_cfg dict for any language choice, including auto-detected ones."""
    if language_choice in LANGUAGE_CONFIG:
        return LANGUAGE_CONFIG[language_choice]
    if detected_langs and language_choice in detected_langs:
        code = detected_langs[language_choice]
        return {"codes": [code], "instruction": "Write your summary in English."}
    return LANGUAGE_CONFIG["English"]


def detect_available_languages(video_url):
    """Return (dropdown update with detected languages, detected_langs dict)."""
    if not video_url or not video_url.strip():
        return gr.update(), {}
    try:
        video_id = extract_video_id(video_url)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        detected = {}
        for t in transcript_list:
            kind = "auto" if t.is_generated else "manual"
            label = f"{t.language} ({kind})"
            detected[label] = t.language_code
        if not detected:
            return gr.update(), {}
        choices = list(detected.keys())
        return gr.update(choices=choices, value=choices[0]), detected
    except Exception as e:
        return gr.update(info=f"⚠️ {e}"), {}


def stream_response(model_choice, messages, system_message=None):
    """Unified streaming dispatcher for all models."""
    try:
        if model_choice == "Claude":
            client = anthropic.Anthropic(timeout=API_TIMEOUT)
            kwargs = dict(model="claude-sonnet-4-6", max_tokens=4096, messages=messages)
            if system_message:
                kwargs["system"] = system_message
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            if model_choice == "DeepSeek":
                client = openai.OpenAI(
                    base_url='https://api.deepseek.com',
                    api_key=os.environ['DEEPSEEK_API_KEY'],
                    timeout=API_TIMEOUT,
                )
                model_id = "deepseek-v4-flash"
            elif model_choice == "DeepSeek Pro":
                client = openai.OpenAI(
                    base_url='https://api.deepseek.com',
                    api_key=os.environ['DEEPSEEK_API_KEY'],
                    timeout=API_TIMEOUT,
                )
                model_id = "deepseek-v4-pro"
            elif model_choice == "OpenAI":
                client = openai.OpenAI(timeout=API_TIMEOUT)
                model_id = "gpt-4o"
            elif model_choice == "Ollama":
                client = openai.OpenAI(base_url='http://localhost:11434/v1', api_key='ollama', timeout=API_TIMEOUT)
                model_id = "cognitivetech/obook_summary:q4_k_m"
            else:
                yield f"⚠️ Unknown model: {model_choice}"
                return
            api_messages = ([{"role": "system", "content": system_message}] if system_message else []) + messages
            stream = client.chat.completions.create(model=model_id, messages=api_messages, stream=True)
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"⚠️ Error ({model_choice}): {e}"


def build_summary_prompt(lang_cfg, transcript_text, use_timestamps=False):
    system_message = (
        "You are an expert at distilling educational content from video transcripts. "
        "Your goal is to capture the essence of what is being taught with precision and fidelity — "
        "preserving the speaker's key ideas, frameworks, and insights without adding, inventing, or embellishing. "
        "Ignore filler words, tangential remarks, and repetition."
    )
    ts_note = (
        "The transcript includes [M:SS](url) timestamp links. "
        "For each Key Concept and Takeaway, include the timestamp link from the nearest relevant line.\n\n"
        if use_timestamps else ""
    )
    depth = _depth_instruction(transcript_text)
    user_message = (
        f"{lang_cfg['instruction']}\n\n"
        f"{ts_note}"
        f"{depth}"
        "Analyze the following YouTube transcript and produce a structured summary that captures its educational essence.\n\n"
        "Use this structure:\n\n"
        "**Core Thesis** (1-2 sentences): What is the central argument or lesson?\n\n"
        "**Key Concepts & Frameworks**: List and briefly explain the main ideas, models, or frameworks introduced.\n\n"
        "**Supporting Evidence & Examples**: Note specific examples, data, stories, or demonstrations used to illustrate the concepts.\n\n"
        "**Actionable Takeaways**: What should the reader understand or be able to do after watching?\n\n"
        "**Notable Quotes** (optional): Include any particularly precise or memorable phrasings from the speaker.\n\n"
        "Rules:\n"
        "- Stay faithful to what was actually said — do not introduce outside knowledge or opinions\n"
        "- Preserve technical terms and domain-specific language exactly as used\n"
        "- If something was unclear in the transcript, reflect that uncertainty rather than guessing\n\n"
        f"Transcript:\n\n{transcript_text}"
    )
    return system_message, user_message


def build_lecture_summary_prompt(lang_cfg, transcript_text, use_timestamps=False):
    system_message = (
        "You are an expert tutor summarising a lecture for a student. "
        "Your goal is to help the student understand and retain the material — not merely archive it. "
        "Explain concepts clearly, as if the student is encountering them for the first time."
    )
    ts_note = (
        "The transcript includes [M:SS](url) timestamp links. "
        "For each Core Concept and Takeaway, include the timestamp link from the nearest relevant line.\n\n"
        if use_timestamps else ""
    )
    depth = _depth_instruction(transcript_text)
    user_message = (
        f"{lang_cfg['instruction']}\n\n"
        f"{ts_note}"
        f"{depth}"
        "Summarise this YouTube lecture for a student who wants to learn from it.\n\n"
        "Use this structure:\n\n"
        "**What You'll Learn** (3–5 bullet objectives): What concrete things will the student be able to understand or do?\n\n"
        "**Core Concepts Explained**: For each major concept, define it clearly, explain the mechanism, and state why it matters. Write for a student encountering it for the first time.\n\n"
        "**Key Examples & Illustrations**: Break down the most important examples used. Explain what each one demonstrates.\n\n"
        "**Frameworks & Models**: Any mental models or structured approaches introduced — described so they can be applied.\n\n"
        "**Takeaways**: The 3–5 most important things to remember from this lecture.\n\n"
        "**Quick Self-Test** (optional): 2–3 questions a student can use to check their understanding.\n\n"
        "Rules:\n"
        "- Prioritise clarity and understanding over brevity\n"
        "- Preserve technical terms exactly as used — then explain them\n"
        "- Stay faithful to what was taught; do not add outside knowledge\n\n"
        f"Transcript:\n\n{transcript_text}"
    )
    return system_message, user_message


def build_technical_summary_prompt(lang_cfg, transcript_text, use_timestamps=False):
    system_message = (
        "You are an expert technical writer distilling coding and software engineering video content. "
        "Produce notes a developer can directly reference and build from — not a high-level summary. "
        "Reproduce code snippets exactly as shown or described. "
        "Reconstruct commands, APIs, and configuration from what was demonstrated. "
        "Format everything for direct use: fenced code blocks, numbered steps, tables."
    )
    ts_note = (
        "The transcript includes [M:SS](url) timestamp links. "
        "For each major step or concept, include the timestamp link from the nearest relevant line.\n\n"
        if use_timestamps else ""
    )
    depth = _depth_instruction(transcript_text)
    user_message = (
        f"{lang_cfg['instruction']}\n\n"
        f"{ts_note}"
        f"{depth}"
        "Produce technical reference notes from this coding/engineering video.\n\n"
        "Use this structure:\n\n"
        "**What This Covers**: 1–2 sentences — what technology, problem, or workflow is demonstrated.\n\n"
        "**Prerequisites**: Tools, versions, libraries, or prior knowledge the speaker assumes.\n\n"
        "**Core Concepts**: For each technical concept, explain it clearly — definition, how it works, why it matters. "
        "Use fenced code blocks to illustrate where helpful.\n\n"
        "**Step-by-Step Walkthrough**: Reproduce the implementation or workflow as numbered steps. "
        "Include all commands, code, and config shown or described. Use fenced code blocks with the correct language tag.\n\n"
        "**Key APIs, Functions & Signatures**: List important functions, classes, CLI flags, or config options — with their parameters and what they do.\n\n"
        "**Architecture & Design**: If a system design is discussed, describe it. Use ASCII diagrams if that helps clarify structure.\n\n"
        "**Gotchas & Caveats**: Bugs, edge cases, version incompatibilities, or warnings the speaker mentioned.\n\n"
        "**Quick Reference**: Compact cheat-sheet of the most-used commands or patterns from this video.\n\n"
        "Rules:\n"
        "- Always use fenced code blocks with the correct language tag (```python, ```bash, ```json, etc.)\n"
        "- Reproduce code as precisely as possible from what was shown or described — never paraphrase code into prose\n"
        "- If a snippet was only partially shown, reconstruct the most likely complete version and mark it [reconstructed]\n"
        "- Preserve exact function names, flag names, and parameter names\n\n"
        f"Transcript:\n\n{transcript_text}"
    )
    return system_message, user_message


def _stream_summary(transcript_text, model_choice, language_choice, note_mode="General", timestamped_transcript=None, detected_langs=None):
    """Yields incrementally accumulated summary string."""
    note_mode = _normalize_mode(note_mode)
    lang_cfg = _resolve_lang_cfg(language_choice, detected_langs)
    effective = timestamped_transcript if timestamped_transcript else transcript_text
    use_ts = bool(timestamped_transcript)
    if note_mode == "Lecture":
        system_message, user_message = build_lecture_summary_prompt(lang_cfg, effective, use_timestamps=use_ts)
    elif note_mode == "Technical":
        system_message, user_message = build_technical_summary_prompt(lang_cfg, effective, use_timestamps=use_ts)
    else:
        system_message, user_message = build_summary_prompt(lang_cfg, effective, use_timestamps=use_ts)
    summary = ""
    for fragment in stream_response(model_choice, [{"role": "user", "content": user_message}], system_message):
        summary += fragment
        yield summary


def format_metadata_card(metadata):
    title = metadata.get('title', 'Unknown')
    channel = metadata.get('channel', '')
    raw_date = metadata.get('publish_date', '')
    view_count = metadata.get('view_count', '')

    date_str = raw_date[:10] if raw_date and raw_date != 'Unknown' else ''
    try:
        views_str = f"{int(view_count):,} views" if view_count and view_count != 'Unknown' else ''
    except (ValueError, TypeError):
        views_str = ''

    meta_parts = [p for p in [channel, date_str, views_str] if p]
    card = f"### {title}"
    if meta_parts:
        card += f"\n*{' · '.join(meta_parts)}*"
    return card


def chat_respond(user_message, history, transcript_text, model_choice, free_mode, language_choice, detected_langs=None):
    if not user_message.strip():
        yield history, ""
        return
    lang_instruction = _resolve_lang_cfg(language_choice, detected_langs)["instruction"]
    if free_mode:
        system_message = (
            f"{lang_instruction}\n\n"
            "You are a helpful assistant. Answer the user's questions thoroughly and freely."
        )
        if transcript_text:
            system_message += f"\n\nFor reference, here is the video transcript:\n\n{transcript_text}"
    else:
        if not transcript_text:
            yield history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "⚠️ Please generate a transcript first before asking questions."},
            ], ""
            return
        system_message = (
            f"{lang_instruction}\n\n"
            "You are a helpful assistant answering follow-up questions about a YouTube video. "
            "Answer concisely based only on the transcript provided. "
            "If the answer is not in the transcript, say so clearly.\n\n"
            f"Transcript:\n\n{transcript_text}"
        )
    messages = list(history) + [{"role": "user", "content": user_message}]
    new_history = list(history) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    for fragment in stream_response(model_choice, messages, system_message):
        new_history[-1]["content"] += fragment
        yield new_history, ""


def gradio_interface(video_url, model_choice, language_choice, note_mode="General", detected_langs=None):
    note_mode = _normalize_mode(note_mode)
    lang_cfg = _resolve_lang_cfg(language_choice, detected_langs)
    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_cfg["codes"])
    except TranscriptsDisabled:
        yield "⚠️ Subtitles are disabled for this video.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False), ""
        return
    except NoTranscriptFound:
        yield f"⚠️ No transcript found for '{language_choice}'.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False), ""
        return
    except VideoUnavailable:
        yield "⚠️ This video is unavailable.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False), ""
        return

    transcript_text = "\n".join(e['text'] for e in entries)
    timestamped_text = _format_timestamped_transcript(entries, video_id)
    metadata_card = format_metadata_card(metadata)
    safe_title = _safe_filename(metadata['title'])
    summary = ""

    for summary in _stream_summary(transcript_text, model_choice, language_choice, note_mode, timestamped_text, detected_langs):
        yield (
            transcript_text, summary, None, None,
            transcript_text, summary, safe_title,
            gr.update(value=metadata_card, visible=True),
            gr.update(open=True),
            timestamped_text,
        )

    transcript_file = f"{safe_title}_transcript.md"
    summary_file = f"{safe_title}_summary.md"
    save_transcript_as_markdown(transcript_text, metadata, transcript_file)
    with open(summary_file, "w") as f:
        f.write(summary)

    yield (
        transcript_text, summary, transcript_file, summary_file,
        transcript_text, summary, safe_title,
        gr.update(value=metadata_card, visible=True),
        gr.update(open=True),
        timestamped_text,
    )


def regenerate_summary(transcript_text, model_choice, language_choice, safe_title, note_mode="General", timestamped_transcript=None, detected_langs=None):
    note_mode = _normalize_mode(note_mode)
    if not transcript_text:
        yield "⚠️ No transcript loaded. Please generate a transcript first.", None, ""
        return
    summary = ""
    for summary in _stream_summary(transcript_text, model_choice, language_choice, note_mode, timestamped_transcript, detected_langs):
        yield summary, None, summary

    filename = f"{safe_title}_summary.md" if safe_title else "summary.md"
    with open(filename, "w") as f:
        f.write(summary)
    yield summary, filename, summary


def reset_all():
    return (
        "", "", None, None,
        "", "", "",
        gr.update(value="", visible=False),
        gr.update(open=False),
        "",
        [],
        gr.update(value="", visible=False),
        None,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False, open=False),
        gr.update(value="", visible=False),
        gr.update(choices=DEFAULT_LANGUAGE_CHOICES, value="English"),
        {},
    )


ZETTELKASTEN_OUTPUT_ROOTS = [
    "/Users/lorenzo/Documents/Youtube Transcript/Source Note",
    "/Users/lorenzo/Library/Mobile Documents/iCloud~md~obsidian/Documents/Lorenzo/2  - Source Material/Video",
]
OBSIDIAN_TAGS_FOLDER = "/Users/lorenzo/Library/Mobile Documents/iCloud~md~obsidian/Documents/Lorenzo/3 - Tags"

_ZK_STOPWORDS = {
    'the','a','an','and','of','in','to','for','with','on','at','from','by','is',
    'are','was','were','be','been','being','have','has','had','do','does','did',
    'will','would','could','should','may','might','must','can','it','its',
    'this','that','these','those','i','you','we','they','he','she',
    'what','how','why','when','where','which','about','into','than','but',
}


def _derive_keyword(title):
    words = re.sub(r'[^\w\s]', ' ', title.lower()).split()
    significant = [w for w in words if w not in _ZK_STOPWORDS and w.isalpha()][:4]
    return '-'.join(significant) if significant else 'video'


def _extract_connections(ref_note_body):
    match = re.search(r'## Connections\s*\n+(.*?)(?:\n---|\Z)', ref_note_body, re.DOTALL)
    if not match:
        return []
    return list(dict.fromkeys(re.findall(r'\[\[([^\]|#\n]+?)(?:\|[^\]])?\]\]', match.group(1))))


def _create_tag_stubs(ref_notes):
    seen = set()
    created = []
    os.makedirs(OBSIDIAN_TAGS_FOLDER, exist_ok=True)
    for _, body in ref_notes:
        for name in _extract_connections(body):
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            tag_path = os.path.join(OBSIDIAN_TAGS_FOLDER, f"{name}.md")
            if not os.path.exists(tag_path):
                with open(tag_path, "w", encoding="utf-8") as f:
                    f.write(f"---\ntype: tag\n---\n\n# {name}\n")
                created.append(name)
    return created


def build_zettelkasten_source_prompt(transcript_text, metadata, video_url, lang_cfg):
    title = metadata.get('title', 'Unknown')
    channel = metadata.get('channel', 'Unknown')
    raw_date = metadata.get('publish_date', '')
    publish_date = raw_date[:10] if raw_date and raw_date != 'Unknown' else 'Unknown'
    today = date.today().isoformat()
    depth = _depth_instruction(transcript_text)

    system_message = (
        "You are a faithful Zettelkasten note-taker. "
        "Capture the source material precisely — do not add outside knowledge, opinions, or interpretations. "
        "Preserve technical terms exactly. Mark speculative or unverified claims with [speculation]."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    user_message = (
        f"Create a Zettelkasten Source Note for this YouTube video.{lang_note}\n\n"
        f"{depth}"
        "CONTENT RULES:\n"
        "- Capture all key points faithfully — do not compress if the source is long\n"
        "- Use H3 subheadings under Key Points if the video has clear segments\n"
        "- Mark speculative or unverified claims with [speculation]\n"
        "- Notable Quotes: use sparingly — only for precise or unusually quotable statements\n"
        "- Critical Framing: briefly note what deserves scrutiny before promoting to a Main Note\n"
        "- Links: fill [[topic]] with 2–4 relevant concepts from the video\n\n"
        "Use this structure:\n\n"
        "```\n"
        f"---\n"
        f"date: {today}\n"
        f"type: source\n"
        f"source-url: {video_url}\n"
        f"speaker: {channel}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"**Speaker / Channel:** {channel}\n"
        f"**URL:** {video_url}\n"
        f"**Date captured:** {today}\n"
        f"**Published:** {publish_date}\n\n"
        f"**Links:** [[topic]] [[topic]]\n\n"
        "---\n\n"
        "## Summary\n\n"
        "2–3 sentences capturing what this source is fundamentally about.\n\n"
        "---\n\n"
        "## Key Points\n\n"
        "Use H3 sections if the video has clear segments. Otherwise use bullet points.\n"
        "Mark speculative claims with [speculation].\n\n"
        "---\n\n"
        "## Notable Quotes\n\n"
        "> \"Exact quote\" — Speaker\n\n"
        "---\n\n"
        "## Critical Framing\n\n"
        "Brief note on what deserves scrutiny or needs verification.\n\n"
        "---\n\n"
        "## See Also\n\n"
        "(Leave empty — reference notes will be listed here.)\n"
        "```\n\n"
        f"TRANSCRIPT:\n\n{transcript_text}"
    )
    return system_message, user_message


def build_zettelkasten_refs_prompt(source_note_text, keyword_date, lang_cfg, transcript_text=None):
    depth = _depth_instruction(transcript_text) if transcript_text else ""
    system_message = (
        "You are a faithful Zettelkasten note-taker extracting Reference Notes from a video. "
        "A Reference Note is a self-contained note about ONE specific concept, technique, or argument — "
        "something with its own name and identity that could exist independently of this source. "
        "It is NOT a re-summary of the source, and it is NOT a sub-section of the source note. "
        "A reader should be able to read a Reference Note without having read the source note and still understand the idea."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    transcript_section = f"\n\nRAW TRANSCRIPT (use this for depth and detail in Key Points):\n\n{transcript_text}" if transcript_text else ""
    user_message = (
        f"Identify concepts from this video that deserve their own Reference Notes, then write each one.{lang_note}\n\n"
        f"{depth}"
        "SELECTION CRITERIA — a good Reference Note candidate:\n"
        "- Has its own name or identity (a framework, technique, concept, argument, phenomenon, or tool)\n"
        "- Is specific enough to be distinct — not just the main topic of the video restated\n"
        "- Could connect to notes on other topics in a personal knowledge base\n"
        "- Is not already fully captured by the source note's Summary section\n\n"
        "ANTI-PATTERNS — do NOT create a Reference Note that:\n"
        "- Is just the video's main topic with a different title\n"
        "- Restates the source note's Summary in bullet form\n"
        "- Duplicates another reference note from the same batch\n"
        "- Is too broad to have a focused title (e.g. 'AI Overview', 'Key Ideas')\n\n"
        "CONTENT RULES:\n"
        "- Generate only as many notes as the content genuinely supports — 0 is acceptable if no distinct sub-concepts exist\n"
        "- Key Points: write 4–8 substantive bullets. Each bullet should be a full, informative sentence — not a phrase. "
        "Draw from the raw transcript for detail; do not compress to vague one-liners\n"
        "- Connections: 2–4 [[wikilinks]] to related concepts beyond this video\n"
        "- Questions: 1–2 things worth verifying or following up on\n\n"
        "OUTPUT RULES:\n"
        "- Output each note one after another with no extra text between them\n"
        "- Each note must start with its own frontmatter block (---)\n"
        "- Titles: 3–6 words, name the concept precisely\n\n"
        "REFERENCE NOTE TEMPLATE (repeat for each note):\n"
        "```\n"
        f"---\n"
        f"date: {date.today().isoformat()}\n"
        f"status: reference\n"
        f"type: reference\n"
        f"source: [[{keyword_date}]]\n"
        f"---\n\n"
        "# Concise Concept Title\n\n"
        "## Key Points\n\n"
        "- Full sentence stating the core claim or definition\n"
        "- Full sentence explaining the mechanism or how it works\n"
        "- Full sentence on the key evidence or argument the speaker gave\n"
        "- Full sentence on an important nuance, caveat, or qualification\n"
        "- Full sentence on the implication or consequence\n\n"
        "---\n\n"
        "## Connections\n\n"
        "[[related topic]] [[related topic]]\n\n"
        "---\n\n"
        "## Questions\n\n"
        "- What would I need to verify this?\n\n"
        "---\n\n"
        "## Source\n\n"
        f"- [[{keyword_date}]]\n"
        "```\n\n"
        f"SOURCE NOTE (for context on what is already captured):\n\n{source_note_text}"
        f"{transcript_section}"
    )
    return system_message, user_message


def build_zettelkasten_lecture_prompt(transcript_text, metadata, video_url, lang_cfg):
    title = metadata.get('title', 'Unknown')
    channel = metadata.get('channel', 'Unknown')
    today = date.today().isoformat()
    depth = _depth_instruction(transcript_text)

    system_message = (
        "You are an expert tutor writing lecture notes for a student. "
        "Prioritise depth and clarity over brevity. "
        "Write as if explaining to a diligent student encountering these ideas for the first time."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    user_message = (
        f"Write lecture notes for a student learning from this video.{lang_note}\n\n"
        f"{depth}"
        "CONTENT RULES:\n"
        "- Prioritise clarity and understanding over brevity — a longer explanation a student can follow is better than a short one they can't\n"
        "- Core Concepts: define each term, explain the mechanism, and state why it matters — write as if the student is encountering it for the first time\n"
        "- Key Examples: explain what each example demonstrates and why the speaker used it\n"
        "- Preserve technical terms exactly as used — then explain them in plain language\n"
        "- Stay faithful to what was taught; do not add outside knowledge\n\n"
        "The template below defines the structure. Apply the rules above to produce rich, educational content.\n\n"
        "TEMPLATE:\n"
        "```\n"
        f"---\n"
        f"date: {today}\n"
        f"type: lecture\n"
        f"source-url: {video_url}\n"
        f"speaker: {channel}\n"
        f"---\n\n"
        f"# {title} — Lecture Notes\n\n"
        f"**Source:** {channel} · {video_url}\n"
        f"**Date:** {today}\n\n"
        "---\n\n"
        "## Learning Objectives\n\n"
        "By the end of this lecture, you should be able to:\n"
        "- \n\n"
        "---\n\n"
        "## Core Concepts\n\n"
        "### Concept Name\n"
        "Full explanation here — define, explain mechanism, state why it matters.\n\n"
        "---\n\n"
        "## Key Examples & Illustrations\n\n"
        "### Example Name\n"
        "What was demonstrated, why this example was used, what it helps the student understand.\n\n"
        "---\n\n"
        "## Frameworks & Models\n\n"
        "Mental models or structured approaches introduced — described clearly enough to apply.\n\n"
        "---\n\n"
        "## Study Questions\n\n"
        "1. \n"
        "2. \n"
        "3. \n\n"
        "---\n\n"
        "## Summary\n\n"
        "3–5 sentences synthesising what was taught and why it matters.\n\n"
        "---\n\n"
        "## See Also\n\n"
        "[[topic]] [[topic]]\n"
        "```\n\n"
        f"TRANSCRIPT:\n\n{transcript_text}"
    )
    return system_message, user_message


def build_zettelkasten_technical_prompt(transcript_text, metadata, video_url, lang_cfg):
    title = metadata.get('title', 'Unknown')
    channel = metadata.get('channel', 'Unknown')
    today = date.today().isoformat()
    depth = _depth_instruction(transcript_text)

    system_message = (
        "You are an expert technical writer creating Zettelkasten reference notes from a coding or engineering video. "
        "Produce notes a developer can directly use: reproduce code exactly, list commands precisely, describe architecture clearly. "
        "Format for direct reference — fenced code blocks, numbered steps, tables."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    user_message = (
        f"Write technical reference notes from this coding/engineering video.{lang_note}\n\n"
        f"{depth}"
        "CONTENT RULES:\n"
        "- Reproduce all code snippets exactly as shown or described — never paraphrase code into prose\n"
        "- Use fenced code blocks with the correct language tag (```python, ```bash, ```json, etc.)\n"
        "- If a snippet was only partially shown, reconstruct the likely complete version and mark it [reconstructed]\n"
        "- Preserve exact function names, flag names, and parameter names\n"
        "- Include every command, config value, and file path mentioned\n"
        "- Gotchas & version notes are first-class content — do not omit them\n\n"
        "The template below defines the structure. Fill every applicable section.\n\n"
        "TEMPLATE:\n"
        "```\n"
        f"---\n"
        f"date: {today}\n"
        f"type: technical\n"
        f"source-url: {video_url}\n"
        f"speaker: {channel}\n"
        f"---\n\n"
        f"# {title} — Technical Notes\n\n"
        f"**Source:** {channel} · {video_url}\n"
        f"**Date:** {today}\n\n"
        "---\n\n"
        "## What This Covers\n\n"
        "1–2 sentences — technology, problem, or workflow demonstrated.\n\n"
        "---\n\n"
        "## Prerequisites\n\n"
        "- Tools, versions, libraries assumed\n\n"
        "---\n\n"
        "## Core Concepts\n\n"
        "### Concept Name\n"
        "Explanation + code block if applicable.\n\n"
        "```language\n"
        "# code here\n"
        "```\n\n"
        "---\n\n"
        "## Step-by-Step Walkthrough\n\n"
        "1. First step\n\n"
        "```bash\n"
        "# command\n"
        "```\n\n"
        "2. Second step\n\n"
        "---\n\n"
        "## Key APIs & Functions\n\n"
        "| Name | Parameters | Description |\n"
        "|------|-----------|-------------|\n"
        "| `fn()` | `param` | What it does |\n\n"
        "---\n\n"
        "## Architecture & Design\n\n"
        "Description or ASCII diagram of the system design.\n\n"
        "---\n\n"
        "## Gotchas & Caveats\n\n"
        "- Known issues, version incompatibilities, warnings\n\n"
        "---\n\n"
        "## Quick Reference\n\n"
        "```bash\n"
        "# most-used commands\n"
        "```\n\n"
        "---\n\n"
        "## See Also\n\n"
        "[[related topic]] [[related topic]]\n"
        "```\n\n"
        f"TRANSCRIPT:\n\n{transcript_text}"
    )
    return system_message, user_message


def preview_zettelkasten_notes(transcript_text, video_url, model_choice, language_choice, note_mode="General", detected_langs=None):
    """Generator yielding (status, source_preview, refs_preview, accordion, zk_state, zk_keyword_input)."""
    note_mode = _normalize_mode(note_mode)
    _nc = gr.update()
    _hidden = gr.update(visible=False, open=False)

    if not transcript_text:
        yield gr.update(value="⚠️ No transcript loaded. Generate a transcript first.", visible=True), _nc, _nc, _hidden, None, _nc
        return

    lang_cfg = _resolve_lang_cfg(language_choice, detected_langs)
    yield gr.update(value="⏳ Fetching video metadata…", visible=True), _nc, _nc, _hidden, None, _nc

    try:
        video_id = extract_video_id(video_url)
        metadata = get_video_metadata(video_id)
    except Exception as e:
        yield gr.update(value=f"⚠️ Could not fetch metadata: {e}", visible=True), _nc, _nc, _hidden, None, _nc
        return

    keyword = _derive_keyword(metadata.get('title', 'video'))
    keyword_date = f"{keyword} {date.today().isoformat()}"

    def _clean_note(text):
        text = text.strip()
        text = re.sub(r'^```\w*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
        text = re.sub(r'^===+\s*\n+', '', text)
        return text.strip()

    if note_mode in ("Lecture", "Technical"):
        mode_label_gen = "lecture note" if note_mode == "Lecture" else "technical note"
        yield gr.update(value=f"⏳ Generating {mode_label_gen} for **{keyword_date}**…", visible=True), _nc, _nc, _hidden, None, _nc

        if note_mode == "Lecture":
            sys_msg, user_msg = build_zettelkasten_lecture_prompt(transcript_text, metadata, video_url, lang_cfg)
        else:
            sys_msg, user_msg = build_zettelkasten_technical_prompt(transcript_text, metadata, video_url, lang_cfg)

        note_body = ""
        for fragment in stream_response(model_choice, [{"role": "user", "content": user_msg}], sys_msg):
            note_body += fragment
        note_body = _clean_note(note_body)
        final_source = note_body
        ref_notes = []
        refs_display = f"*{mode_label_gen.capitalize()} — no separate reference notes generated.*"
    else:
        yield gr.update(value=f"⏳ Generating source note for **{keyword_date}**…", visible=True), _nc, _nc, _hidden, None, _nc
        sys_msg, user_msg = build_zettelkasten_source_prompt(transcript_text, metadata, video_url, lang_cfg)
        source_note = ""
        for fragment in stream_response(model_choice, [{"role": "user", "content": user_msg}], sys_msg):
            source_note += fragment
        source_note = _clean_note(source_note)

        yield gr.update(value="⏳ Generating reference notes…", visible=True), _nc, _nc, _hidden, None, _nc
        sys_msg2, user_msg2 = build_zettelkasten_refs_prompt(source_note, keyword_date, lang_cfg, transcript_text=transcript_text)
        refs_raw = ""
        for fragment in stream_response(model_choice, [{"role": "user", "content": user_msg2}], sys_msg2):
            refs_raw += fragment

        ref_parts = re.split(r'(?m)(?=^---\ndate:)', refs_raw.strip())
        ref_notes = []
        for part in ref_parts:
            part = _clean_note(part)
            if not part.startswith('---'):
                continue
            title_match = re.search(r'^# (.+)$', part, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
                ref_notes.append((title, part))

        see_also_links = "\n".join(f"- [[ref - {t}]]" for t, _ in ref_notes)
        if see_also_links and '## See Also' in source_note:
            parts = source_note.split('## See Also', 1)
            final_source = parts[0] + f"## See Also\n\n{see_also_links}\n"
        else:
            final_source = source_note
        refs_display = "\n\n---\n\n".join(_strip_frontmatter(body) for _, body in ref_notes) if ref_notes else ""

    count_label = (
        "lecture note" if note_mode == "Lecture"
        else "technical note" if note_mode == "Technical"
        else f"{len(ref_notes)} reference notes"
    )
    yield (
        gr.update(value=f"✅ Preview ready ({count_label}) — review below, then click **Save to Zettelkasten**.", visible=True),
        gr.update(value=_strip_frontmatter(final_source)),
        gr.update(value=refs_display),
        gr.update(visible=True, open=True),
        {"keyword_date": keyword_date, "source_note": final_source, "ref_notes": ref_notes, "note_mode": note_mode},
        gr.update(value=keyword_date, visible=True),
    )


def save_zettelkasten_notes(zk_data, custom_keyword=None):
    if not zk_data:
        return gr.update(value="⚠️ No preview to save. Click Preview first.", visible=True)

    keyword_date = custom_keyword.strip() if custom_keyword and custom_keyword.strip() else zk_data["keyword_date"]
    source_note = zk_data["source_note"]
    ref_notes = zk_data["ref_notes"]
    note_mode = zk_data.get("note_mode", "General")

    if note_mode == "Lecture":
        main_filename = f"{keyword_date} (lecture).md"
    elif note_mode == "Technical":
        main_filename = f"{keyword_date} (technical).md"
    else:
        main_filename = f"{keyword_date}.md"

    saved_paths = []
    try:
        for root in ZETTELKASTEN_OUTPUT_ROOTS:
            folder = os.path.join(root, keyword_date)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, main_filename), "w", encoding="utf-8") as f:
                f.write(source_note)
            for title, body in ref_notes:
                with open(os.path.join(folder, f"ref - {title}.md"), "w", encoding="utf-8") as f:
                    f.write(body)
            saved_paths.append(folder)
    except Exception as e:
        return gr.update(value=f"⚠️ Error saving files: {e}", visible=True)

    created_tags = _create_tag_stubs(ref_notes)

    paths_list = "\n".join(f"- `{p}`" for p in saved_paths)
    tags_section = (
        f"\n\n**Tag stubs created ({len(created_tags)}):** "
        + ", ".join(f"`{t}`" for t in created_tags)
        if created_tags else ""
    )
    if note_mode in ("Lecture", "Technical"):
        mode_word = "Lecture" if note_mode == "Lecture" else "Technical"
        summary = (
            f"✅ **{mode_word} note saved**\n\n"
            f"**Folders:**\n{paths_list}\n\n"
            f"**File:** `{main_filename}`"
            f"{tags_section}"
        )
    else:
        ref_list = "\n".join(f"  - `ref - {t}.md`" for t, _ in ref_notes)
        summary = (
            f"✅ **Saved** ({len(ref_notes)} reference notes)\n\n"
            f"**Folders:**\n{paths_list}\n\n"
            f"**Source Note:** `{main_filename}`\n\n"
            f"**Reference Notes:**\n{ref_list}"
            f"{tags_section}"
        )
    return gr.update(value=summary, visible=True)


READING_CSS = """
#summary-output .prose, #transcript-output .prose, #chat-panel .prose {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 17px !important;
    line-height: 1.8 !important;
    color: #1a1a1a !important;
    letter-spacing: 0.01em !important;
}
#summary-output .prose p, #transcript-output .prose p,
#summary-output .prose li, #transcript-output .prose li,
#chat-panel .prose p, #chat-panel .prose li {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 17px !important;
    line-height: 1.8 !important;
    margin-bottom: 0.75em !important;
}
#summary-output .prose h1, #summary-output .prose h2, #summary-output .prose h3,
#transcript-output .prose h1, #transcript-output .prose h2, #transcript-output .prose h3,
#chat-panel .prose h1, #chat-panel .prose h2, #chat-panel .prose h3 {
    font-family: Georgia, 'Times New Roman', serif !important;
    letter-spacing: -0.01em !important;
}
#summary-output .prose strong, #transcript-output .prose strong, #chat-panel .prose strong {
    font-weight: 700 !important;
    color: #111 !important;
}
#metadata-card {
    padding: 10px 16px;
    border-left: 3px solid #6366f1;
    background: rgba(99, 102, 241, 0.05);
    border-radius: 4px;
    margin-bottom: 4px;
}
#zettelkasten-status {
    padding: 10px 16px;
    border-left: 3px solid #10b981;
    background: rgba(16, 185, 129, 0.05);
    border-radius: 4px;
    margin-top: 8px;
}
#zk-source-preview .prose, #zk-refs-preview .prose {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 16px !important;
    line-height: 1.75 !important;
    color: #1a1a1a !important;
}
#zk-source-preview .prose p, #zk-refs-preview .prose p,
#zk-source-preview .prose li, #zk-refs-preview .prose li {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 16px !important;
    line-height: 1.75 !important;
    margin-bottom: 0.6em !important;
}
#zk-source-preview .prose h1, #zk-source-preview .prose h2, #zk-source-preview .prose h3,
#zk-refs-preview .prose h1, #zk-refs-preview .prose h2, #zk-refs-preview .prose h3 {
    font-family: Georgia, 'Times New Roman', serif !important;
}
"""


def main():
    with gr.Blocks(theme=gr.themes.Soft(), title="YouTube Transcript Extractor", css=READING_CSS) as demo:
        gr.Markdown("# YouTube Transcript Extractor")
        gr.Markdown("Extract and summarize YouTube videos using AI — paste a URL, pick a model and language, and go.")

        transcript_state = gr.State("")
        summary_state = gr.State("")
        title_state = gr.State("")
        timestamped_state = gr.State("")
        detected_langs_state = gr.State({})

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                video_url_input = gr.Textbox(
                    label="YouTube Video URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["DeepSeek", "DeepSeek Pro", "Claude", "OpenAI", "Ollama"],
                    label="Model",
                    value="DeepSeek",
                )
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    choices=DEFAULT_LANGUAGE_CHOICES,
                    label="Subtitle Language",
                    value="English",
                )
            with gr.Column(scale=0, min_width=90):
                detect_langs_btn = gr.Button("🔍 Detect", size="sm", variant="secondary")

        with gr.Row():
            submit_button = gr.Button("Generate Transcript & Summary", variant="primary", size="lg", scale=4)
            mode_dropdown = gr.Dropdown(
                choices=["General", "Lecture", "Technical"],
                label="Mode",
                value="General",
                scale=1,
            )
            reset_button = gr.Button("Reset", variant="secondary", size="lg", scale=1)

        metadata_output = gr.Markdown(visible=False, elem_id="metadata-card")

        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_output = gr.Markdown(height=520, show_label=False, elem_id="summary-output")
                with gr.Row():
                    copy_summary_btn = gr.Button("📋 Copy", size="sm", variant="secondary", scale=1)
                    regenerate_button = gr.Button("↺ Regenerate", size="sm", variant="secondary", scale=1)
                    download_summary_button = gr.File(label="Download", scale=2)

            with gr.Tab("Transcript"):
                transcript_output = gr.Markdown(height=520, show_label=False, elem_id="transcript-output")
                with gr.Row():
                    copy_transcript_btn = gr.Button("📋 Copy", size="sm", variant="secondary", scale=1)
                    download_transcript_button = gr.File(label="Download", scale=3)

        _copy_hidden_summary = gr.Textbox(visible=False, elem_id="copy-hidden-summary")
        _copy_hidden_transcript = gr.Textbox(visible=False, elem_id="copy-hidden-transcript")

        with gr.Row():
            zettelkasten_btn = gr.Button("📓 Preview Zettelkasten Notes", variant="secondary", size="lg")
        zettelkasten_status = gr.Markdown(visible=False, elem_id="zettelkasten-status")
        zk_state = gr.State(None)

        with gr.Accordion("Zettelkasten Preview", open=True, visible=False) as zk_preview_accordion:
            with gr.Tabs():
                with gr.Tab("Source Note"):
                    zk_source_preview = gr.Markdown(height=420, show_label=False, elem_id="zk-source-preview")
                with gr.Tab("Reference Notes"):
                    zk_refs_preview = gr.Markdown(height=420, show_label=False, elem_id="zk-refs-preview")
            with gr.Row():
                zk_keyword_input = gr.Textbox(label="Folder name (editable)", scale=3, visible=False)
                save_zk_btn = gr.Button("💾 Save to Zettelkasten", variant="primary", size="lg", scale=2)

        gr.Markdown("---")
        with gr.Accordion("Ask a follow-up question", open=False) as chat_accordion:
            chatbot = gr.Chatbot(type="messages", height=350, show_label=False, elem_id="chat-panel")
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask anything about the video...",
                    show_label=False,
                    scale=5,
                )
                chat_button = gr.Button("Ask", scale=1, variant="secondary")
            free_mode_toggle = gr.Checkbox(label="Answer freely (not limited to transcript)", value=False)

        gen_outputs = [
            transcript_output, summary_output,
            download_transcript_button, download_summary_button,
            transcript_state, summary_state, title_state,
            metadata_output, chat_accordion,
            timestamped_state,
        ]

        detect_langs_btn.click(
            detect_available_languages,
            inputs=[video_url_input],
            outputs=[language_dropdown, detected_langs_state],
        )
        submit_button.click(
            gradio_interface,
            inputs=[video_url_input, model_dropdown, language_dropdown, mode_dropdown, detected_langs_state],
            outputs=gen_outputs,
        )
        reset_button.click(
            reset_all,
            outputs=gen_outputs + [
                chatbot, zettelkasten_status, zk_state,
                zk_source_preview, zk_refs_preview, zk_preview_accordion,
                zk_keyword_input, language_dropdown, detected_langs_state,
            ],
        )
        regenerate_button.click(
            regenerate_summary,
            inputs=[transcript_state, model_dropdown, language_dropdown, title_state, mode_dropdown, timestamped_state, detected_langs_state],
            outputs=[summary_output, download_summary_button, summary_state],
        )
        copy_summary_btn.click(
            fn=lambda t: t, inputs=[summary_state], outputs=[_copy_hidden_summary],
        ).then(
            fn=None, inputs=[], outputs=[],
            js="() => { const el = document.querySelector('#copy-hidden-summary textarea'); if (el) navigator.clipboard.writeText(el.value).catch(() => { el.focus(); el.select(); document.execCommand('copy'); }); }",
        )
        copy_transcript_btn.click(
            fn=lambda t: t, inputs=[transcript_state], outputs=[_copy_hidden_transcript],
        ).then(
            fn=None, inputs=[], outputs=[],
            js="() => { const el = document.querySelector('#copy-hidden-transcript textarea'); if (el) navigator.clipboard.writeText(el.value).catch(() => { el.focus(); el.select(); document.execCommand('copy'); }); }",
        )
        chat_button.click(
            chat_respond,
            inputs=[chat_input, chatbot, transcript_state, model_dropdown, free_mode_toggle, language_dropdown, detected_langs_state],
            outputs=[chatbot, chat_input],
        )
        chat_input.submit(
            chat_respond,
            inputs=[chat_input, chatbot, transcript_state, model_dropdown, free_mode_toggle, language_dropdown, detected_langs_state],
            outputs=[chatbot, chat_input],
        )
        zettelkasten_btn.click(
            preview_zettelkasten_notes,
            inputs=[transcript_state, video_url_input, model_dropdown, language_dropdown, mode_dropdown, detected_langs_state],
            outputs=[zettelkasten_status, zk_source_preview, zk_refs_preview, zk_preview_accordion, zk_state, zk_keyword_input],
        )
        save_zk_btn.click(
            save_zettelkasten_notes,
            inputs=[zk_state, zk_keyword_input],
            outputs=[zettelkasten_status],
        )

    demo.launch()


if __name__ == "__main__":
    main()

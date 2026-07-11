import gradio as gr
import os
import re
import openai
import anthropic
from datetime import date
from dotenv import load_dotenv

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


def stream_response(model_choice, messages, system_message=None):
    """Unified streaming dispatcher for all models. messages: list of {role, content}."""
    try:
        if model_choice == "Claude":
            client = anthropic.Anthropic(timeout=API_TIMEOUT)
            kwargs = dict(model="claude-sonnet-4-6", max_tokens=2000, messages=messages)
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


def build_summary_prompt(lang_cfg, transcript_text):
    system_message = (
        "You are an expert at distilling educational content from video transcripts. "
        "Your goal is to capture the essence of what is being taught with precision and fidelity — "
        "preserving the speaker's key ideas, frameworks, and insights without adding, inventing, or embellishing. "
        "Ignore filler words, tangential remarks, and repetition."
    )
    user_message = (
        f"{lang_cfg['instruction']}\n\n"
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


def _stream_summary(transcript_text, model_choice, language_choice):
    """Yields incrementally accumulated summary string."""
    lang_cfg = LANGUAGE_CONFIG[language_choice]
    system_message, user_message = build_summary_prompt(lang_cfg, transcript_text)
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


def chat_respond(user_message, history, transcript_text, model_choice, free_mode, language_choice):
    if not user_message.strip():
        yield history, ""
        return
    lang_instruction = LANGUAGE_CONFIG[language_choice]["instruction"]
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


def gradio_interface(video_url, model_choice, language_choice):
    lang_cfg = LANGUAGE_CONFIG[language_choice]
    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)

    try:
        transcript_text = get_youtube_transcript(video_id, languages=lang_cfg["codes"])
    except TranscriptsDisabled:
        yield "⚠️ Subtitles are disabled for this video.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False)
        return
    except NoTranscriptFound:
        yield f"⚠️ No {language_choice} transcript found for this video.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False)
        return
    except VideoUnavailable:
        yield "⚠️ This video is unavailable.", "", None, None, "", "", "", gr.update(visible=False), gr.update(open=False)
        return

    metadata_card = format_metadata_card(metadata)
    safe_title = _safe_filename(metadata['title'])
    summary = ""

    for summary in _stream_summary(transcript_text, model_choice, language_choice):
        yield (
            transcript_text, summary, None, None,
            transcript_text, summary, safe_title,
            gr.update(value=metadata_card, visible=True),
            gr.update(open=True),
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
    )


def regenerate_summary(transcript_text, model_choice, language_choice, safe_title):
    if not transcript_text:
        yield "⚠️ No transcript loaded. Please generate a transcript first.", None, ""
        return
    summary = ""
    for summary in _stream_summary(transcript_text, model_choice, language_choice):
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
        [],
        gr.update(value="", visible=False),
        None,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False, open=False),
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
    """Return unique [[wikilink]] targets from the ## Connections section."""
    match = re.search(r'## Connections\s*\n+(.*?)(?:\n---|\Z)', ref_note_body, re.DOTALL)
    if not match:
        return []
    return list(dict.fromkeys(re.findall(r'\[\[([^\]|#\n]+?)(?:\|[^\]])?\]\]', match.group(1))))


def _create_tag_stubs(ref_notes):
    """Create stub notes in OBSIDIAN_TAGS_FOLDER for new connections. Returns list of created names."""
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

    system_message = (
        "You are a faithful Zettelkasten note-taker. "
        "Your job is to capture the source material precisely using the provided template. "
        "Do not add outside knowledge, opinions, or interpretations beyond what the speaker said. "
        "Preserve technical terms exactly. Mark speculative or unverified claims with [speculation]."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    user_message = (
        f"Create a Source Note for this YouTube video using EXACTLY the template below. "
        f"Fill in every section with content from the transcript.{lang_note}\n\n"
        "TEMPLATE:\n"
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
        "(Use sparingly — only for precise or unusually quotable statements.)\n\n"
        "---\n\n"
        "## Critical Framing\n\n"
        "Brief note on what deserves scrutiny or needs verification before promoting to a Main Note.\n\n"
        "---\n\n"
        "## See Also\n\n"
        "(Leave this section empty — reference notes will be listed here separately.)\n"
        "```\n\n"
        f"TRANSCRIPT:\n\n{transcript_text}"
    )
    return system_message, user_message


def build_zettelkasten_refs_prompt(source_note_text, keyword_date, lang_cfg):
    system_message = (
        "You are a faithful Zettelkasten note-taker. "
        "Your job is to distill 2–5 self-contained ideas from a source into Reference Notes. "
        "Each note captures one idea exactly as the speaker presented it — not your synthesis."
    )
    lang_note = f"\n\n{lang_cfg['instruction']}" if lang_cfg.get('instruction') else ""
    user_message = (
        f"Based on the Source Note below, identify 2–5 ideas worth capturing as Reference Notes.{lang_note}\n\n"
        "RULES:\n"
        "- Start each reference note with this exact delimiter on its own line: ===REF: <Concise Idea Title>===\n"
        "- Follow each delimiter immediately with the note content using the template below\n"
        "- Titles should be concise (3–6 words), capturing the core idea\n"
        "- Do not add any text between notes except the delimiter line\n\n"
        "REFERENCE NOTE TEMPLATE (repeat for each note):\n"
        "```\n"
        f"---\n"
        f"date: {date.today().isoformat()}\n"
        f"status: reference\n"
        f"type: reference\n"
        f"source: [[{keyword_date}]]\n"
        f"---\n\n"
        "# <Concise Idea Title>\n\n"
        "## Key Points\n\n"
        "- <Core claim or definition>\n"
        "- <Key mechanism, argument, or evidence>\n"
        "- <Important nuance or caveat>\n"
        "- <Implication or consequence>\n"
        "- <Additional point if relevant>\n\n"
        "---\n\n"
        "## Connections\n\n"
        "[[related topic]] [[related topic]]\n\n"
        "---\n\n"
        "## Questions\n\n"
        "- What would I need to verify this?\n"
        "- Which existing notes does this connect to?\n\n"
        "---\n\n"
        "## Source\n\n"
        f"- [[{keyword_date}]]\n"
        "```\n\n"
        f"SOURCE NOTE:\n\n{source_note_text}"
    )
    return system_message, user_message


def preview_zettelkasten_notes(transcript_text, video_url, model_choice, language_choice):
    """Generator yielding (status, source_preview, refs_preview, accordion, zk_state)."""
    _nc = gr.update()
    _hidden = gr.update(visible=False, open=False)

    if not transcript_text:
        yield gr.update(value="⚠️ No transcript loaded. Generate a transcript first.", visible=True), _nc, _nc, _hidden, None
        return

    lang_cfg = LANGUAGE_CONFIG[language_choice]
    yield gr.update(value="⏳ Fetching video metadata…", visible=True), _nc, _nc, _hidden, None

    try:
        video_id = extract_video_id(video_url)
        metadata = get_video_metadata(video_id)
    except Exception as e:
        yield gr.update(value=f"⚠️ Could not fetch metadata: {e}", visible=True), _nc, _nc, _hidden, None
        return

    keyword = _derive_keyword(metadata.get('title', 'video'))
    keyword_date = f"{keyword} {date.today().isoformat()}"

    yield gr.update(value=f"⏳ Generating source note for **{keyword_date}**…", visible=True), _nc, _nc, _hidden, None
    sys_msg, user_msg = build_zettelkasten_source_prompt(transcript_text, metadata, video_url, lang_cfg)
    source_note = ""
    for fragment in stream_response(model_choice, [{"role": "user", "content": user_msg}], sys_msg):
        source_note += fragment

    yield gr.update(value="⏳ Generating reference notes…", visible=True), _nc, _nc, _hidden, None
    sys_msg2, user_msg2 = build_zettelkasten_refs_prompt(source_note, keyword_date, lang_cfg)
    refs_raw = ""
    for fragment in stream_response(model_choice, [{"role": "user", "content": user_msg2}], sys_msg2):
        refs_raw += fragment

    def _clean_note(text):
        text = text.strip()
        text = re.sub(r'^```\w*\n', '', text)   # strip opening code fence
        text = re.sub(r'\n```\s*$', '', text)   # strip closing code fence
        text = re.sub(r'^===+\s*\n+', '', text) # strip stray === delimiter lines
        return text.strip()

    source_note = _clean_note(source_note)

    ref_blocks = re.split(r'===REF:\s*(.+?)===', refs_raw)
    ref_notes = []
    for i in range(1, len(ref_blocks) - 1, 2):
        title = ref_blocks[i].strip()
        body = _clean_note(ref_blocks[i + 1])
        ref_notes.append((title, body))

    see_also_links = "\n".join(f"- [[ref - {t}]]" for t, _ in ref_notes)
    final_source = source_note.replace(
        "## See Also\n\n(Leave this section empty — reference notes will be listed here separately.)",
        f"## See Also\n\n{see_also_links}",
    )

    refs_preview = "\n\n---\n\n".join(body for _, body in ref_notes)

    yield (
        gr.update(value=f"✅ Preview ready ({len(ref_notes)} reference notes) — review below, then click **Save to Zettelkasten**.", visible=True),
        gr.update(value=final_source),
        gr.update(value=refs_preview),
        gr.update(visible=True, open=True),
        {"keyword_date": keyword_date, "source_note": final_source, "ref_notes": ref_notes},
    )


def save_zettelkasten_notes(zk_data):
    if not zk_data:
        return gr.update(value="⚠️ No preview to save. Click Preview first.", visible=True)

    keyword_date = zk_data["keyword_date"]
    source_note = zk_data["source_note"]
    ref_notes = zk_data["ref_notes"]

    saved_paths = []
    try:
        for root in ZETTELKASTEN_OUTPUT_ROOTS:
            folder = os.path.join(root, keyword_date)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"{keyword_date}.md"), "w", encoding="utf-8") as f:
                f.write(source_note)
            for title, body in ref_notes:
                with open(os.path.join(folder, f"ref - {title}.md"), "w", encoding="utf-8") as f:
                    f.write(body)
            saved_paths.append(folder)
    except Exception as e:
        return gr.update(value=f"⚠️ Error saving files: {e}", visible=True)

    created_tags = _create_tag_stubs(ref_notes)

    ref_list = "\n".join(f"  - `ref - {t}.md`" for t, _ in ref_notes)
    paths_list = "\n".join(f"- `{p}`" for p in saved_paths)
    tags_section = (
        f"\n\n**Tag stubs created ({len(created_tags)}):** "
        + ", ".join(f"`{t}`" for t in created_tags)
        if created_tags else ""
    )
    return gr.update(
        value=(
            f"✅ **Saved** ({len(ref_notes)} reference notes)\n\n"
            f"**Folders:**\n{paths_list}\n\n"
            f"**Source Note:** `{keyword_date}.md`\n\n"
            f"**Reference Notes:**\n{ref_list}"
            f"{tags_section}"
        ),
        visible=True,
    )


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
"""


def main():
    with gr.Blocks(theme=gr.themes.Soft(), title="YouTube Transcript Extractor", css=READING_CSS) as demo:
        gr.Markdown("# YouTube Transcript Extractor")
        gr.Markdown("Extract and summarize YouTube videos using AI — paste a URL, pick a model and language, and go.")

        transcript_state = gr.State("")
        summary_state = gr.State("")
        title_state = gr.State("")

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
                    choices=["English", "Chinese"],
                    label="Language",
                    value="English",
                )

        with gr.Row():
            submit_button = gr.Button("Generate Transcript & Summary", variant="primary", size="lg", scale=4)
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
                    zk_source_preview = gr.Markdown(height=420, show_label=False)
                with gr.Tab("Reference Notes"):
                    zk_refs_preview = gr.Markdown(height=420, show_label=False)
            with gr.Row():
                save_zk_btn = gr.Button("💾 Save to Zettelkasten", variant="primary", size="lg")

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
        ]

        submit_button.click(
            gradio_interface,
            inputs=[video_url_input, model_dropdown, language_dropdown],
            outputs=gen_outputs,
        )
        reset_button.click(reset_all, outputs=gen_outputs + [chatbot, zettelkasten_status, zk_state, zk_source_preview, zk_refs_preview, zk_preview_accordion])
        regenerate_button.click(
            regenerate_summary,
            inputs=[transcript_state, model_dropdown, language_dropdown, title_state],
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
            inputs=[chat_input, chatbot, transcript_state, model_dropdown, free_mode_toggle, language_dropdown],
            outputs=[chatbot, chat_input],
        )
        chat_input.submit(
            chat_respond,
            inputs=[chat_input, chatbot, transcript_state, model_dropdown, free_mode_toggle, language_dropdown],
            outputs=[chatbot, chat_input],
        )
        zettelkasten_btn.click(
            preview_zettelkasten_notes,
            inputs=[transcript_state, video_url_input, model_dropdown, language_dropdown],
            outputs=[zettelkasten_status, zk_source_preview, zk_refs_preview, zk_preview_accordion, zk_state],
        )
        save_zk_btn.click(
            save_zettelkasten_notes,
            inputs=[zk_state],
            outputs=[zettelkasten_status],
        )

    demo.launch()


if __name__ == "__main__":
    main()

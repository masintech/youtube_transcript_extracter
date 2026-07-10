import gradio as gr
import os
import openai
import anthropic
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


def get_ollama_response(model, user_message, system_message=None):
    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.append({"role": "user", "content": user_message})

    try:
        ollama_via_openai = openai.OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        stream = ollama_via_openai.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            yield fragment
    except Exception as e:
        yield f"Error generating response: {e}"


def get_openai_response(model, user_message, system_message=None):
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        stream = openai.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            yield fragment
    except Exception as e:
        yield f"Error generating response: {e}"


def get_anthropic_claude_response(model, user_message, system_message=None):
    try:
        claude = anthropic.Anthropic()
        kwargs = dict(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": user_message}],
        )
        if system_message:
            kwargs["system"] = system_message
        with claude.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Error generating response: {e}"


def get_deepseek_response(model, user_message, system_message=None):
    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.append({"role": "user", "content": user_message})

    try:
        deepseek_via_openai = openai.OpenAI(
            base_url='https://api.deepseek.com',
            api_key=os.environ['DEEPSEEK_API_KEY'],
        )
        stream = deepseek_via_openai.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            yield fragment
    except Exception as e:
        yield f"Error generating response: {e}"


LANGUAGE_CONFIG = {
    "English": {"codes": ["en"],           "instruction": "Write your summary in English."},
    "Chinese": {"codes": ["zh-Hant", "zh"], "instruction": "Write your summary in Traditional Chinese (繁體中文)."},
}


def gradio_interface(video_url, model_choice, language_choice):
    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)
    lang_cfg = LANGUAGE_CONFIG[language_choice]
    try:
        transcript_text = get_youtube_transcript(video_id, languages=lang_cfg["codes"])
    except TranscriptsDisabled:
        yield "⚠️ Subtitles are disabled for this video.", "", None, None
        return
    except NoTranscriptFound:
        yield f"⚠️ No {language_choice} transcript found for this video.", "", None, None
        return
    except VideoUnavailable:
        yield "⚠️ This video is unavailable.", "", None, None
        return

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
    summary = f"Rewrite of the video '{metadata['title']}':\n\n"

    if model_choice == "DeepSeek":
        for fragment in get_deepseek_response("deepseek-v4-flash", user_message, system_message):
            summary += fragment
            yield transcript_text, summary, None, None
    elif model_choice == "Claude":
        for fragment in get_anthropic_claude_response("claude-sonnet-4-6", user_message, system_message):
            summary += fragment
            yield transcript_text, summary, None, None
    elif model_choice == "Ollama":
        for fragment in get_ollama_response("cognitivetech/obook_summary:q4_k_m", user_message, system_message):
            summary += fragment
            yield transcript_text, summary, None, None
    elif model_choice == "OpenAI":
        for fragment in get_openai_response("gpt-4o", user_message, system_message):
            summary += fragment
            yield transcript_text, summary, None, None

    safe_title = _safe_filename(metadata['title'])
    transcript_file = f"{safe_title}_transcript.md"
    summary_file = f"{safe_title}_summary.md"
    save_transcript_as_markdown(transcript_text, metadata, transcript_file)
    with open(summary_file, "w") as f:
        f.write(summary)

    yield transcript_text, summary, transcript_file, summary_file


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# YouTube Transcript Extractor")

        with gr.Row():
            video_url_input = gr.Textbox(
                label="YouTube Video URL",
                placeholder="Enter YouTube video URL here...",
            )
            model_dropdown = gr.Dropdown(
                choices=["DeepSeek", "Claude", "Ollama", "OpenAI"],
                label="Model",
                value="DeepSeek",
            )
            language_dropdown = gr.Dropdown(
                choices=["English", "Chinese"],
                label="Transcript Language",
                value="English",
            )

        with gr.Row():
            transcript_output = gr.Markdown(label="Transcript")
            summary_output = gr.Markdown(label="Summary")

        with gr.Row():
            download_transcript_button = gr.File(label="Download Transcript")
            download_summary_button = gr.File(label="Download Summary")

        submit_button = gr.Button("Generate Transcript and Summary")

        submit_button.click(
            gradio_interface,
            inputs=[video_url_input, model_dropdown, language_dropdown],
            outputs=[transcript_output, summary_output, download_transcript_button, download_summary_button],
        )

    demo.launch()


if __name__ == "__main__":
    main()

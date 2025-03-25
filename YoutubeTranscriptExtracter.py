import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

def get_video_metadata(video_url):
    ydl_opts = {
        "quiet": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    
    metadata = {
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "description": info.get("description"),
        "publish_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "dislike_count": info.get("dislike_count"),  # May not be available
        "duration": info.get("duration"),  # In seconds
    }
    
    return metadata

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([entry["text"] for entry in transcript])
    except Exception as e:
        return f"Error: {e}"

def save_transcript_as_markdown(transcript, metadata, output_file="transcript.md"):
    """
    Save the transcript to a Markdown file with a formatted structure, including metadata.

    Args:
        transcript (str): The transcript text.
        metadata (dict): The metadata of the video.
        output_file (str): The name of the output Markdown file (optional).
    """
    try:
        with open(output_file, "w") as md_file:
            # Write the title
            md_file.write(f"# {metadata['title']}\n\n")
            md_file.write("---\n\n")
            
            # Write metadata
            md_file.write("## Video Information\n\n")
            md_file.write(f"- **Channel**: {metadata['channel']}\n")
            md_file.write(f"- **Description**: {metadata['description']}\n")
            md_file.write(f"- **Publish Date**: {metadata['publish_date']}\n")
            md_file.write(f"- **View Count**: {metadata['view_count']}\n")
            md_file.write("---\n\n")
            
            # Write the transcript
            md_file.write("## Transcript\n\n")
            md_file.write(transcript)
        return output_file
    except Exception as e:
        return f"Error saving transcript: {e}"

def process_video(video_url):
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    metadata = get_video_metadata(video_url)
    transcript_text = get_youtube_transcript(video_id)
    output_file = f"{metadata['title']}.md"
    save_transcript_as_markdown(transcript_text, metadata, output_file)
    return output_file

# Gradio Interface
def gradio_interface(video_url):
    output_file = process_video(video_url)
    return output_file

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Transcript Extractor")
    with gr.Row():
        video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
        download_button = gr.File(label="Download Markdown File")
    submit_button = gr.Button("Generate Transcript")
    
    submit_button.click(gradio_interface, inputs=video_url_input, outputs=download_button)

# demo.launch(share=True)
demo.launch(share=True)

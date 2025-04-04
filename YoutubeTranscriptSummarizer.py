import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import os
from openai import OpenAI

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


def check_ollama():
    # Test if ollama is working, if not, run command "ollama serve"
    try:
        import ollama
        print("Ollama is working!")
    except ImportError:
        print("Ollama is not working. Run the command 'ollama serve' to start the server.")

def get_transcript(video_url):
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    transcript_text = get_youtube_transcript(video_id)
    return transcript_text


def read_message_from_file(file_path):
    """
    Reads a message from a file.

    Args:
        file_path (str): Path to the file containing the message.

    Returns:
        str: The content of the file.
    """
    if not os.path.exists(file_path):
        return "Error: File does not exist."
    with open(file_path, "r") as file:
        message = file.read()
        if message.startswith("Error"):
            print(message)
            return
        return message


def write_response_to_file(response, output_file):
    """
    Writes the response to a file.

    Args:
        response (str): The response content to write.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as file:
        file.write(response)

# In this use case, user message should contain user prompt and transcript text
def get_ollama_response(model, user_message, system_message=None):
    """
    Interacts with the Ollama API to generate a response based on the user message and model.
    Args:
        user_message (str): The user's message or prompt.
        model (str): The model to use for the chat.
        system_message (str, optional): An optional system message.
    Returns:
        str: The generated response from the model.
    """
    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.append({"role": "user", "content": user_message})

    # MODEL = "llama3.2"

    # if not installed, !ollama pull deepseek-r1:7b
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    response = ollama_via_openai.chat.completions.create(
        model=model,
        messages=api_messages
    )
    response_content = response.choices[0].message.content
    return response_content

def gradio_interface(video_url):
    """
    Processes the YouTube video URL, extracts the transcript, and generates a summary.

    Args:
        video_url (str): The YouTube video URL.

    Returns:
        tuple: The transcript, the summary, and file paths for downloading.
    """
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    metadata = get_video_metadata(video_url)
    transcript_text = get_youtube_transcript(video_id)
    
    user_message = f"Summarize the following transcript:\n\n{transcript_text}"
    # Generate a summary (placeholder logic, replace with actual summarization logic)
    summary = f"Summary of the video '{metadata['title']}':\n\n"
    summary += get_ollama_response("cognitivetech/obook_summary:q4_k_m", user_message, system_message="You are a helpful assistant that summarizes transcripts.")

    
    # Save transcript and summary to files
    transcript_file = f"{metadata['title']}_transcript.md"
    summary_file = f"{metadata['title']}_summary.md"
    save_transcript_as_markdown(transcript_text, metadata, transcript_file)
    write_response_to_file(summary, summary_file)
    
    return transcript_text, summary, transcript_file, summary_file

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# YouTube Transcript Extractor")
        
        with gr.Row():
            video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
        
        with gr.Row():
            # transcript_output = gr.Textbox(label="Transcript", lines=10, interactive=False)
            # summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
            transcript_output = gr.Markdown(label="Transcript")
            summary_output = gr.Markdown(label="Summary")
        
        with gr.Row():
            download_transcript_button = gr.File(label="Download Transcript")
            download_summary_button = gr.File(label="Download Summary")
        
        submit_button = gr.Button("Generate Transcript and Summary")
        
        # Connect the interface components
        submit_button.click(
            gradio_interface,
            inputs=video_url_input,
            outputs=[transcript_output, summary_output, download_transcript_button, download_summary_button]
        )

    demo.launch()


if __name__ == "__main__":
    main()

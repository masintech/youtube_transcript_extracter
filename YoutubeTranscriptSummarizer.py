import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import os
import openai
import anthropic
from dotenv import load_dotenv


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')


def get_video_metadata(video_id):
    youtube = build('youtube', 'v3', developerKey=os.environ['GOOGLE_API_KEY'])
    request = youtube.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()
    
    if "items" in response and len(response["items"]) > 0:
        video_data = response["items"][0]
        metadata = {
            "title": video_data["snippet"]["title"],
            "channel": video_data["snippet"]["channelTitle"],
            "description": video_data["snippet"]["description"],
            "publish_date": video_data["snippet"]["publishedAt"],
            "view_count": video_data["statistics"]["viewCount"]
        }
        return metadata
    else:
        return {
            "title": "Unknown",
            "channel": "Unknown",
            "description": "Unknown",
            "publish_date": "Unknown",
            "view_count": "Unknown"
        }

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
    metadata = get_video_metadata(video_id)  # Pass the correct video_id here
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
    try:
        ollama_via_openai = openai.OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        stream = ollama_via_openai.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            reply += fragment
            yield fragment
        return reply
    except Exception as e:
        return f"Error generating response: {e}"

def get_openai_response(model, user_message, system_message=None):
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        stream = openai.chat.completions.create(model=model, messages=messages, stream=True)
        reply = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            reply += fragment
            # print(fragment, end='', flush=True)
            yield fragment
        return reply
    except Exception as e:
        return f"Error generating response: {e}"

def get_anthopic_claude_response(model, user_message, system_message=None):
    try:
        claude = anthropic.Anthropic()
        if system_message:
            result = claude.messages.stream(
                model=model,
                max_tokens=2000,
                system=system_message,
                messages=[{"role": "user", "content": user_message}],)
        else:
            result = claude.messages.stream(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": user_message}],)
        
        reply = ""
        with result as stream:
            for text in stream.text_stream:
                reply += text
                print(text, end="", flush=True)
                
        return reply
    except Exception as e:
        return f"Error generating response: {e}"

def get_deepseek_response(model, user_message, system_message=None):
    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.append({"role": "user", "content": user_message})

    try:
        deepseek_via_openai = openai.OpenAI(base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY'])
        stream = deepseek_via_openai.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            reply += fragment
            yield fragment
        return reply
    except Exception as e:
        return f"Error generating response: {e}"

def gradio_interface(video_url, model_choice):
    """
    gradio interface function to process the video URL and model choice.
    Args:
        video_url (str): The YouTube video URL.
        model_choice (str): The model choice ("Claude" or "OpenAI").
    Returns:
        tuple: The transcript text and summary.
    """
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    metadata = get_video_metadata(video_id)
    transcript_text = get_youtube_transcript(video_id)
    
    user_message = f"Summarize the following transcript:\n\n{transcript_text}"
    summary = f"Summary of the video '{metadata['title']}':\n\n"
    
    # Display the summary in real-time and collect all yielded values
    # choose to call get_openai_response or get_anthopic_claude_response based on the variable
    if model_choice == "DeepSeek":
        for fragment in get_deepseek_response("deepseek-chat", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
            summary += fragment
            yield transcript_text, summary, None, None  # Update the summary_output in real-time
      
    elif model_choice == "Claude":
        # for fragment in get_anthopic_claude_response("claude-3-5-sonnet-20240620", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
        for fragment in get_anthopic_claude_response("claude-3-7-sonnet-20250219", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
            summary += fragment
            yield transcript_text, summary, None, None  # Update the summary_output in real-time
    elif model_choice == "Ollama":
        for fragment in get_ollama_response("cognitivetech/obook_summary:q4_k_m", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
            summary += fragment
            yield transcript_text, summary, None, None  # Update the summary_output in real-time
    elif model_choice == "OpenAI":
        # for fragment in get_openai_response("gpt-4o-mini", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
        for fragment in get_openai_response("gpt-4o", user_message, system_message="You are a helpful assistant that summarizes transcripts."):
            summary += fragment
            yield transcript_text, summary, None, None  # Update the summary_output in real-time
    
    # Save transcript and summary to files
    transcript_file = f"{metadata['title']}_transcript.md"
    summary_file = f"{metadata['title']}_summary.md"
    save_transcript_as_markdown(transcript_text, metadata, transcript_file)
    write_response_to_file(summary, summary_file)
    
    yield transcript_text, summary, transcript_file, summary_file  # Final output with file paths

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# YouTube Transcript Extractor")
        
        with gr.Row():
            video_url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
            model_dropdown = gr.Dropdown(
                choices=["DeepSeek", "Claude", "Ollama", "OpenAI",], label="Model", value="DeepSeek"
            )
        
        with gr.Row():
            transcript_output = gr.Markdown(label="Transcript")
            summary_output = gr.Markdown(label="Summary")
        
        with gr.Row():
            download_transcript_button = gr.File(label="Download Transcript")
            download_summary_button = gr.File(label="Download Summary")
        
        submit_button = gr.Button("Generate Transcript and Summary")
        
        # Connect the interface components
        submit_button.click(
            gradio_interface,
            inputs=[video_url_input, model_dropdown],
            outputs=[transcript_output, summary_output, download_transcript_button, download_summary_button]
        )

    demo.launch(share=True)


if __name__ == "__main__":
    main()

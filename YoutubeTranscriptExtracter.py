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
            md_file.write("---\n\n")
            
            # Write the transcript
            md_file.write("## Transcript\n\n")
            md_file.write(transcript)
        print(f"Transcript saved as Markdown file: {output_file}")
    except Exception as e:
        print(f"Error saving transcript: {e}")

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    
    # Get metadata and transcript
    metadata = get_video_metadata(video_url)
    transcript_text = get_youtube_transcript(video_id)
    
    # Save the transcript as a Markdown file with metadata
    save_transcript_as_markdown(transcript_text, metadata, output_file=f"{metadata['title']}.md")

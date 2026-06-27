from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import os
import re
import sys
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(video_url):
    parsed = urlparse(video_url)
    if parsed.hostname == 'youtu.be':
        return parsed.path.lstrip('/')
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        if parsed.path.startswith('/shorts/'):
            return parsed.path.split('/shorts/')[1].split('/')[0]
    raise ValueError(f"Could not extract video ID from URL: {video_url}")


def get_video_metadata(video_id):
    youtube = build('youtube', 'v3', developerKey=os.getenv('GOOGLE_API_KEY'))
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
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return "\n".join([entry["text"] for entry in transcript])

def save_transcript_as_markdown(transcript, metadata, output_file="transcript.md"):
    with open(output_file, "w") as md_file:
        md_file.write(f"# {metadata['title']}\n\n")
        md_file.write("---\n\n")
        md_file.write("## Video Information\n\n")
        md_file.write(f"- **Channel**: {metadata['channel']}\n")
        md_file.write(f"- **Description**: {metadata['description']}\n")
        md_file.write(f"- **Publish Date**: {metadata['publish_date']}\n")
        md_file.write(f"- **View Count**: {metadata['view_count']}\n")
        md_file.write("---\n\n")
        md_file.write("## Transcript\n\n")
        md_file.write(transcript)
    return output_file

def _safe_filename(title):
    return re.sub(r'[^\w\s-]', '', title).strip() or "transcript"

def process_video(video_url):
    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)
    transcript_text = get_youtube_transcript(video_id)
    output_file = f"{_safe_filename(metadata['title'])}.md"
    save_transcript_as_markdown(transcript_text, metadata, output_file)
    return output_file


def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    return get_youtube_transcript(video_id)


# CLI entry point
def print_usage():
    print("Usage:")
    print("  python YoutubeTranscriptionExtrator.py {URL}         # Print transcript to terminal")
    print("  python YoutubeTranscriptionExtrator.py {URL} -l      # Output transcript to file {title}.md")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    video_url = sys.argv[1]
    output_to_file = len(sys.argv) > 2 and sys.argv[2] == "-l"

    try:
        if output_to_file:
            output_file = process_video(video_url)
            print(f"Transcript saved to {output_file}")
        else:
            print(get_transcript(video_url))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import argparse
import os
import re
import sys
import pyperclip
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
    print("Fetching metadata...", file=sys.stderr)
    youtube = build('youtube', 'v3', developerKey=os.getenv('GOOGLE_API_KEY'))
    request = youtube.videos().list(part="snippet,statistics", id=video_id)
    response = request.execute()

    if "items" in response and len(response["items"]) > 0:
        video_data = response["items"][0]
        return {
            "title": video_data["snippet"]["title"],
            "channel": video_data["snippet"]["channelTitle"],
            "description": video_data["snippet"]["description"],
            "publish_date": video_data["snippet"]["publishedAt"],
            "view_count": video_data["statistics"]["viewCount"]
        }
    return {
        "title": "Unknown",
        "channel": "Unknown",
        "description": "Unknown",
        "publish_date": "Unknown",
        "view_count": "Unknown"
    }


def get_youtube_transcript(video_id, languages=None):
    print("Downloading transcript...", file=sys.stderr)
    kwargs = {"languages": languages} if languages else {}
    transcript = YouTubeTranscriptApi.get_transcript(video_id, **kwargs)
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


def process_video(video_url, languages=None, output_file=None):
    video_id = extract_video_id(video_url)
    metadata = get_video_metadata(video_id)
    transcript_text = get_youtube_transcript(video_id, languages)
    if output_file is None:
        output_file = f"{_safe_filename(metadata['title'])}.md"
    save_transcript_as_markdown(transcript_text, metadata, output_file)
    return output_file


def get_transcript(video_url, languages=None):
    video_id = extract_video_id(video_url)
    return get_youtube_transcript(video_id, languages)


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcripts from YouTube videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s https://youtube.com/watch?v=ID\n"
            "  %(prog)s https://youtu.be/ID -l\n"
            "  %(prog)s https://youtu.be/ID -l --out my_notes.md\n"
            "  %(prog)s https://youtu.be/ID --lang fr es\n"
        )
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-l", "--save",
        action="store_true",
        help="Save transcript to a Markdown file instead of printing"
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Output file path (implies -l, default: <video title>.md)"
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        metavar="LANG",
        help="Preferred transcript language(s) in order of priority (e.g. --lang fr en)"
    )
    parser.add_argument(
        "-c", "--copy",
        action="store_true",
        help="Copy transcript to clipboard"
    )
    args = parser.parse_args()

    save_to_file = args.save or args.out is not None

    try:
        if save_to_file:
            output_file = process_video(args.url, languages=args.lang, output_file=args.out)
            print(f"Transcript saved to {output_file}", file=sys.stderr)
        else:
            transcript = get_transcript(args.url, languages=args.lang)
            if args.copy:
                pyperclip.copy(transcript)
                print("Transcript copied to clipboard.", file=sys.stderr)
            else:
                print(transcript)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([entry["text"] for entry in transcript])
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID
    transcript_text = get_youtube_transcript(video_id)
    
    print("\nTranscript:\n", transcript_text)

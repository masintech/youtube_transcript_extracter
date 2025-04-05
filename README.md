# youtube_transcript_extracter

## Overview
This project extracts YouTube video transcripts, summarizes them using various AI models, and saves the results in Markdown files.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd youtube_transcript_extracter
```

### 2. Install Dependencies
Make sure you have Python installed. Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project directory and add the following keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_openai_api_key`, `your_anthropic_api_key`, `your_deepseek_api_key`, and `your_google_api_key` with your actual API keys.

### 4. Pull Required Ollama Model
If you plan to use the Ollama model, ensure the Ollama server is running and pull the required model:
```bash
ollama pull cognitivetech/obook_summary:q4_k_m
```

### 5. Run the Application
Launch the Gradio interface:
```bash
python YoutubeTranscriptSummarizer.py
```

## Usage
1. Enter the YouTube video URL in the input box.
2. Select the AI model for summarization (DeepSeek, Claude, Ollama, or OpenAI).
3. Click "Generate Transcript and Summary" to process the video.
4. Download the transcript and summary files if needed.

## Notes
- Ensure all API keys are valid and have sufficient quota.
- The Ollama server must be running locally for the Ollama model to work. Use the command `ollama serve` to start the server.
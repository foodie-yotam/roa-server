# 🖥️ Server Microservice

Flask API server that orchestrates voice processing and agent communication.

## Tech Stack

- **Framework**: Flask
- **Language**: Python 3.11+
- **Deployment**: Railway

## Files

- `web_server.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `railway.json` - Railway deployment config
- `.env.example` - Environment variable template

## Environment Variables

```bash
OPENAI_API_KEY=          # For Whisper STT
ELEVENLABS_API_KEY=      # For TTS
ELEVENLABS_VOICE_ID=     # Voice to use
LANGGRAPH_URL=           # LangGraph Cloud endpoint
LANGSMITH_API_KEY=       # LangGraph authentication
PORT=5001                # Server port (Railway sets this)
```

## Endpoints

### `GET /`
Serves the HTML interface

### `POST /process_voice`
Handles voice input and returns audio response

**Request:**
- `audio`: WAV file (multipart/form-data)
- `user_id`: User identifier for conversation thread

**Response:**
```json
{
  "transcript": "What recipes are available?",
  "response": "We have 3 recipes: Caprese Salad...",
  "audio_url": "/audio/12345"
}
```

### `GET /audio/<audio_id>`
Streams generated audio response

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run server
python web_server.py
```

Server runs on http://localhost:5001

## Deploy to Railway

1. Push to GitHub
2. Connect Railway to your repo
3. Add environment variables in Railway dashboard
4. Deploy!

Railway will automatically:
- Detect Python
- Install dependencies
- Run `python web_server.py`
- Assign a public URL

## Architecture

```
Browser
  ↓ POST /process_voice (audio file)
Flask Server
  ↓ Transcribe
OpenAI Whisper API
  ↓ Text
Flask Server
  ↓ Call agent
LangGraph Cloud
  ↓ Response text
Flask Server
  ↓ Generate speech
ElevenLabs API
  ↓ Audio
Browser (plays audio)
```

## Dependencies

- `flask` - Web framework
- `flask-cors` - CORS support
- `openai` - Whisper STT
- `elevenlabs` - TTS
- `langgraph-sdk` - LangGraph Cloud client
- `python-dotenv` - Environment variables

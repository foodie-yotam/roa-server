#!/usr/bin/env python3
"""Flask web server for voice agent"""
import os
import io
import uuid
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from langchain_core.messages import HumanMessage, convert_to_messages

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# LangGraph setup
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://localhost:8123")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GRAPH_NAME = "agent"  # Deployed graph name

# Create LangGraph client
langgraph_client = get_sync_client(url=LANGGRAPH_URL, api_key=LANGSMITH_API_KEY)
print(f"🔗 Connecting to LangGraph at: {LANGGRAPH_URL}")
print(f"🔑 Using API key: {LANGSMITH_API_KEY[:20] if LANGSMITH_API_KEY else 'None'}...")

# Thread cache: maps user_id to thread_id
thread_cache = {}

# Store audio responses temporarily
audio_cache = {}

def get_or_create_thread(user_id: str) -> str:
    """Get existing thread_id for user or create a new one"""
    if user_id not in thread_cache:
        # Create a new thread for this user
        thread = langgraph_client.threads.create()
        thread_cache[user_id] = thread["thread_id"]
        print(f"Created new thread for user {user_id}: {thread['thread_id']}")
    return thread_cache[user_id]

def call_agent(thread_id: str, message: str) -> str:
    """Call the LangGraph agent and return response"""
    input_data = {"messages": [{"role": "user", "content": message}]}
    response_text = ""
    
    try:
        print(f"📡 Calling LangGraph: thread={thread_id}, graph={GRAPH_NAME}")
        for chunk in langgraph_client.runs.stream(
            thread_id,
            GRAPH_NAME,
            input=input_data,
            stream_mode="updates"
        ):
            print(f"📦 Chunk: {chunk.data if hasattr(chunk, 'data') else chunk}")
            if hasattr(chunk, 'data') and chunk.data and "run_id" not in chunk.data:
                for key, value in chunk.data.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                response_text = last_msg["content"]
                                print(f"✅ Got response: {response_text[:100]}...")
    except Exception as e:
        print(f"❌ LangGraph error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if not response_text:
        response_text = "I apologize, but I couldn't process your request. Please try again."
    
    return response_text

@app.route('/')
def index():
    # Serve HTML from interface folder
    interface_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interface', 'web_voice.html')
    print(f"🌐 Serving HTML from: {interface_path}")
    print(f"📁 File exists: {os.path.exists(interface_path)}")
    if os.path.exists(interface_path):
        print(f"✅ Serving web_voice.html")
        return send_file(interface_path)
    else:
        print(f"❌ File not found at {interface_path}")
        return jsonify({"error": "HTML file not found", "path": interface_path}), 404

@app.route('/process_voice_text_only', methods=['POST'])
def process_voice_text_only():
    """Process voice without ElevenLabs (text response only)"""
    try:
        # Get audio file and user_id
        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'web-user')
        
        # Save to BytesIO
        audio_bytes = io.BytesIO(audio_file.read())
        audio_bytes.name = "audio.wav"
        
        # Transcribe with Whisper
        print("Transcribing...")
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes
        )
        transcript_text = transcription.text
        print(f"Transcript: {transcript_text}")
        
        # Get or create thread for this user
        thread_id = get_or_create_thread(user_id)
        
        # Send to agent with thread_id for conversation memory
        print(f"Sending to agent (thread: {thread_id})...")
        
        # Prepare input for LangGraph
        input_data = {"messages": [{"role": "user", "content": transcript_text}]}
        
        # Call LangGraph Cloud using SDK
        response_text = ""
        for chunk in langgraph_client.runs.stream(
            thread_id,
            GRAPH_NAME,
            input=input_data,
            stream_mode="updates"
        ):
            if chunk.data and "run_id" not in chunk.data:
                for key, value in chunk.data.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                response_text = last_msg["content"]
        
        if not response_text:
            response_text = "I apologize, but I couldn't process your request. Please try again."
        
        print(f"Agent response: {response_text}")
        
        return jsonify({
            'transcript': transcript_text,
            'response': response_text
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Simple chat endpoint without Whisper (uses browser speech recognition)"""
    try:
        data = request.json
        message = data.get('message', '')
        user_id = data.get('user_id', 'web-user')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"Message: {message}")
        
        # Get or create thread for this user
        thread_id = get_or_create_thread(user_id)
        
        # Send to agent with thread_id for conversation memory
        print(f"Sending to agent (thread: {thread_id})...")
        
        # Prepare input for LangGraph
        input_data = {"messages": [{"role": "user", "content": message}]}
        
        # Call LangGraph Cloud using SDK
        response_text = ""
        for chunk in langgraph_client.runs.stream(
            thread_id,
            GRAPH_NAME,
            input=input_data,
            stream_mode="updates"
        ):
            if chunk.data and "run_id" not in chunk.data:
                for key, value in chunk.data.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                response_text = last_msg["content"]
        
        if not response_text:
            response_text = "I apologize, but I couldn't process your request. Please try again."
        
        print(f"Agent response: {response_text}")
        
        # Generate speech with ElevenLabs
        print("Generating speech...")
        cleaned_text = response_text.replace("**", "")
        
        audio_response = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=cleaned_text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        
        # Save audio to cache
        audio_id = str(hash(response_text))
        audio_data = b''.join(audio_response)
        audio_cache[audio_id] = audio_data
        
        return jsonify({
            'response': response_text,
            'audio_url': f'/audio/{audio_id}'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        # Get audio file and user_id
        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'web-user')
        
        # Save to BytesIO
        audio_bytes = io.BytesIO(audio_file.read())
        audio_bytes.name = "audio.wav"
        
        # Transcribe with Whisper
        print("Transcribing...")
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes
        )
        transcript_text = transcription.text
        print(f"Transcript: {transcript_text}")
        
        # Get or create thread for this user
        thread_id = get_or_create_thread(user_id)
        
        # Call agent
        response_text = call_agent(thread_id, transcript_text)
        
        # Generate speech with ElevenLabs
        print("Generating speech...")
        cleaned_text = response_text.replace("**", "")
        
        audio_response = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=cleaned_text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        
        # Save audio to cache
        audio_id = str(hash(response_text))
        audio_data = b''.join(audio_response)
        audio_cache[audio_id] = audio_data
        
        return jsonify({
            'transcript': transcript_text,
            'response': response_text,
            'audio_url': f'/audio/{audio_id}'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<audio_id>')
def get_audio(audio_id):
    if audio_id in audio_cache:
        return send_file(
            io.BytesIO(audio_cache[audio_id]),
            mimetype='audio/mpeg'
        )
    return "Audio not found", 404

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    print(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

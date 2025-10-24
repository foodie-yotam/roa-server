#!/usr/bin/env python3
"""Flask web server for voice agent"""
import os
import io
import uuid
import json
from typing import Generator
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph_sdk import get_sync_client
from langchain_core.messages import HumanMessage, convert_to_messages

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Server URL for absolute audio URLs (Railway provides RAILWAY_PUBLIC_DOMAIN automatically)
railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
SERVER_URL = f"https://{railway_domain}" if railway_domain else os.getenv("SERVER_URL", "http://localhost:5001")

# LangGraph setup
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://localhost:8123")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GRAPH_NAME = "agent"  # Deployed graph name

# Create LangGraph client
langgraph_client = get_sync_client(url=LANGGRAPH_URL, api_key=LANGSMITH_API_KEY)
print(f"🌐 Server URL: {SERVER_URL}")
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
                    # Only capture assistant responses (not tool calls)
                    if key == "assistant" and isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                content = last_msg.get("content", "")
                                if content and isinstance(content, str) and content.strip():
                                    response_text = content
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
    return jsonify({
        "service": "ROA Voice API",
        "status": "running",
        "endpoints": {
            "/chat": "POST - Text chat with agent",
            "/process_voice": "POST - Voice input processing",
            "/process_voice_text_only": "POST - Voice to text only",
            "/audio/<filename>": "GET - Retrieve audio response",
            "/v1/chat/completions": "POST - OpenAI-compatible endpoint (for ElevenLabs)"
        }
    })

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
        print("=" * 60)
        print("💬 [Backend] /chat endpoint hit")
        
        data = request.json
        message = data.get('message', '')
        user_id = data.get('user_id', 'web-user')
        
        print(f"📝 [Backend] Message: {message}")
        print(f"👤 [Backend] User ID: {user_id}")
        
        if not message:
            print("❌ [Backend] No message provided")
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create thread for this user
        thread_id = get_or_create_thread(user_id)
        print(f"🧵 [Backend] Thread ID: {thread_id}")
        
        # Send to agent with thread_id for conversation memory
        print(f"🚀 [Backend] Sending to LangGraph agent...")
        
        # Prepare input for LangGraph
        input_data = {"messages": [{"role": "user", "content": message}]}
        print(f"📦 [Backend] Input data: {input_data}")
        
        # Call LangGraph Cloud using SDK
        response_text = ""
        chunk_count = 0
        for chunk in langgraph_client.runs.stream(
            thread_id,
            GRAPH_NAME,
            input=input_data,
            stream_mode="updates"
        ):
            chunk_count += 1
            print(f"📨 [Backend] Chunk {chunk_count} received")
            if chunk.data and "run_id" not in chunk.data:
                for key, value in chunk.data.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                response_text = last_msg["content"]
                                print(f"✅ [Backend] Response extracted: {response_text[:100]}...")
        
        print(f"📊 [Backend] Total chunks processed: {chunk_count}")
        
        if not response_text:
            response_text = "I apologize, but I couldn't process your request. Please try again."
            print("⚠️ [Backend] No response from agent, using fallback")
        
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
            'audio_url': f'{SERVER_URL}/audio/{audio_id}'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        print("=" * 60)
        print("🎙️ [Backend] /process_voice endpoint hit")
        
        # Get audio file and user_id
        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'web-user')
        
        print(f"📁 [Backend] Audio file received: {audio_file.filename}")
        print(f"👤 [Backend] User ID: {user_id}")
        
        # Save to BytesIO
        audio_bytes = io.BytesIO(audio_file.read())
        audio_bytes.name = "audio.wav"
        audio_size = len(audio_bytes.getvalue())
        print(f"📊 [Backend] Audio size: {audio_size} bytes")
        
        # Transcribe with Whisper
        print("🎧 [Backend] Starting Whisper transcription...")
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes
        )
        transcript_text = transcription.text
        print(f"✅ [Backend] Transcript: {transcript_text}")
        
        # Get or create thread for this user
        thread_id = get_or_create_thread(user_id)
        print(f"🧵 [Backend] Thread ID: {thread_id}")
        
        # Call agent
        print("🤖 [Backend] Calling LangGraph agent...")
        response_text = call_agent(thread_id, transcript_text)
        print(f"💬 [Backend] Agent response: {response_text[:100]}...")
        
        # Generate speech with ElevenLabs
        print("🔊 [Backend] Generating speech with ElevenLabs...")
        cleaned_text = response_text.replace("**", "")
        print(f"📝 [Backend] Cleaned text length: {len(cleaned_text)} chars")
        
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
        print(f"💾 [Backend] Audio cached with ID: {audio_id}, size: {len(audio_data)} bytes")
        
        print(f"✅ [Backend] Request completed successfully")
        print("=" * 60)
        
        return jsonify({
            'transcript': transcript_text,
            'response': response_text,
            'audio_url': f'{SERVER_URL}/audio/{audio_id}'
        })
        
    except Exception as e:
        print("=" * 60)
        print(f"💥 [Backend] Exception in /process_voice: {e}")
        import traceback
        print(f"📍 [Backend] Traceback:\n{traceback.format_exc()}")
        print("=" * 60)
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<audio_id>')
def get_audio(audio_id):
    if audio_id in audio_cache:
        return send_file(
            io.BytesIO(audio_cache[audio_id]),
            mimetype='audio/mpeg'
        )
    return "Audio not found", 404


# ============================================================================
# OpenAI-Compatible Endpoint for ElevenLabs Conversational AI Integration
# ============================================================================

def stream_openai_response(thread_id: str, message: str) -> Generator[str, None, None]:
    """
    Stream LangGraph agent responses in OpenAI-compatible format.
    This endpoint is used by ElevenLabs Conversational AI Custom LLM integration.
    """
    input_data = {"messages": [{"role": "user", "content": message}]}
    has_content = False
    accumulated_text = ""
    
    try:
        print(f"[OpenAI Endpoint] Calling LangGraph: thread={thread_id}")
        
        # Stream from LangGraph
        for chunk in langgraph_client.runs.stream(
            thread_id,
            GRAPH_NAME,
            input=input_data,
            stream_mode="updates"
        ):
            if hasattr(chunk, 'data') and chunk.data and "run_id" not in chunk.data:
                for key, value in chunk.data.items():
                    # Only capture assistant responses (not tool calls)
                    if key == "assistant" and isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and "content" in last_msg:
                                content = last_msg.get("content", "")
                                # Only send if there's actual text content
                                if content and isinstance(content, str) and content.strip():
                                    response_text = content
                                    accumulated_text = response_text
                                    
                                    # Stream in OpenAI format
                                    chunk_data = {
                                        "id": f"chatcmpl-{thread_id}",
                                        "object": "chat.completion.chunk",
                                        "created": 1234567890,
                                        "model": "langgraph-agent",
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": response_text if not has_content else "",
                                                "role": "assistant" if not has_content else None
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    has_content = True
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send final chunk with finish_reason
        if has_content:
            final_chunk = {
                "id": f"chatcmpl-{thread_id}",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "langgraph-agent",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        
        # OpenAI completion signal
        yield "data: [DONE]\n\n"
        print(f"[OpenAI Endpoint] Response complete: {accumulated_text[:100]}...")
        
    except Exception as e:
        print(f"[OpenAI Endpoint] Error: {e}")
        error_chunk = {
            "id": f"chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "langgraph-agent",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    """
    OpenAI-compatible chat completions endpoint for ElevenLabs integration.
    
    This endpoint allows ElevenLabs Conversational AI to use our LangGraph agent
    as a Custom LLM by providing an OpenAI-compatible API interface.
    
    Expected request format:
    {
        "messages": [{"role": "user", "content": "message text"}],
        "model": "langgraph-agent",
        "stream": true,
        "user": "user_id"
    }
    """
    try:
        data = request.get_json()
        print(f"[OpenAI Endpoint] Received request: {len(data.get('messages', []))} messages")
        
        # Extract messages
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content')
                break
        
        if not user_message:
            return jsonify({"error": "No user message found"}), 400
        
        # Get or create thread for this user
        user_id = data.get('user', 'elevenlabs-user')
        thread_id = get_or_create_thread(user_id)
        
        print(f"[OpenAI Endpoint] User: {user_id}, Thread: {thread_id}, Message: {user_message[:50]}...")
        
        # Stream response in OpenAI format
        return Response(
            stream_with_context(stream_openai_response(thread_id, user_message)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        print(f"[OpenAI Endpoint] Exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    print(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

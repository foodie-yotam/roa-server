#!/usr/bin/env python3
"""Flask web server for voice agent"""
import os
import io
import uuid
import json
from typing import Generator, Optional, Tuple
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
# OpenAI and ElevenLabs imports removed - using ElevenLabs widget directly
from langgraph_sdk import get_sync_client
from langchain_core.messages import HumanMessage, convert_to_messages
import hashlib

load_dotenv()

app = Flask(__name__)
CORS(app)

# Clients removed - ElevenLabs widget handles STT/TTS directly

# Server URL for absolute audio URLs (Railway provides RAILWAY_PUBLIC_DOMAIN automatically)
railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
SERVER_URL = f"https://{railway_domain}" if railway_domain else os.getenv("SERVER_URL", "http://localhost:5001")

# LangGraph setup - Support dev, staging, and production
LANGGRAPH_URL_DEV = os.getenv("LANGGRAPH_URL_DEV", "https://roa-voice-dev-6d8ceb540dee59cebd2fc361aa316dec.us.langgraph.app")
LANGGRAPH_URL_STAGING = os.getenv("LANGGRAPH_URL_STAGING", "https://roa-voice-staging-66a6f6ec95d9546995a4f3352ad05df2.us.langgraph.app")
LANGGRAPH_URL_PROD = os.getenv("LANGGRAPH_URL_PROD", "https://roa-voice-prod-59235014c2ad5a5f830e9e124171824f.us.langgraph.app")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GRAPH_NAME = "agent"  # Deployed graph name

# Create LangGraph clients for all environments
langgraph_client_dev = get_sync_client(url=LANGGRAPH_URL_DEV, api_key=LANGSMITH_API_KEY)
langgraph_client_staging = get_sync_client(url=LANGGRAPH_URL_STAGING, api_key=LANGSMITH_API_KEY)
langgraph_client_prod = get_sync_client(url=LANGGRAPH_URL_PROD, api_key=LANGSMITH_API_KEY)

print(f"🌐 Server URL: {SERVER_URL}")
print(f"🔗 Dev LangGraph: {LANGGRAPH_URL_DEV}")
print(f"🔗 Staging LangGraph: {LANGGRAPH_URL_STAGING}")
print(f"🔗 Production LangGraph: {LANGGRAPH_URL_PROD}")
print(f"🔑 Using API key: {LANGSMITH_API_KEY[:20] if LANGSMITH_API_KEY else 'None'}...")

# Audio cache removed - ElevenLabs widget handles TTS

# Thread cache: maps our deterministic thread_id to LangGraph's actual thread_id
# This is lightweight - just stores the mapping, LangGraph stores the actual conversation
thread_id_cache = {}


def _generate_thread_id() -> str:
    """Generate a random thread ID for fallback cases."""
    return str(uuid.uuid4())


def _hash_caller_id(caller_id: str, environment: str = "prod") -> str:
    """Convert caller_id to a deterministic thread identifier.
    
    This ensures:
    1. Each unique caller gets their own isolated thread
    2. Same caller always maps to same thread (conversation continuity within session)
    3. Dev, staging, and production threads are separate
    4. No server-side caching needed - LangGraph manages thread persistence
    
    Args:
        caller_id: Unique identifier for the caller (from ElevenLabs)
        environment: Environment name ("dev", "staging", or "prod")
    
    Returns:
        Deterministic thread identifier that can be used directly with LangGraph
    """
    # Add environment prefix to ensure dev/staging/prod separation
    env_prefix = environment
    
    # Create deterministic hash from caller_id
    # Using SHA256 for consistent, collision-resistant hashing
    combined = f"{env_prefix}:{caller_id}"
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    
    # Return first 16 chars of hex digest for readability
    # This gives us 64 bits of entropy (extremely low collision probability)
    return f"{env_prefix}-{hash_obj.hexdigest()[:16]}"


def get_langgraph_client(environment: str = "prod"):
    """Get the appropriate LangGraph client based on environment.
    
    Args:
        environment: Environment name ("dev", "staging", or "prod")
    
    Returns:
        LangGraph client for the specified environment
    """
    if environment == "dev":
        return langgraph_client_dev
    elif environment == "staging":
        return langgraph_client_staging
    else:
        return langgraph_client_prod


def get_thread_id_for_caller(caller_id: Optional[str] = None, environment: str = "prod") -> str:
    """Get or create LangGraph thread for a caller.
    
    Creates a deterministic mapping: caller_id → LangGraph thread_id
    The thread is actually created in LangGraph on first use.
    
    Args:
        caller_id: Unique caller identifier (from ElevenLabs convo_id or caller_id)
        environment: Environment name ("dev", "staging", or "prod")
    
    Returns:
        LangGraph thread ID to use with API calls
    """
    # Generate fallback if no caller_id provided
    if not caller_id:
        caller_id = _generate_thread_id()
        print(f"⚠️  No caller_id provided, generated random: {caller_id}")
    
    # Create deterministic cache key
    cache_key = _hash_caller_id(caller_id, environment)
    
    # Check if we already have a LangGraph thread for this caller
    if cache_key not in thread_id_cache:
        # Create actual thread in LangGraph
        client = get_langgraph_client(environment)
        thread = client.threads.create()
        langgraph_thread_id = thread["thread_id"]
        thread_id_cache[cache_key] = langgraph_thread_id
        print(f"✨ Created {environment} thread for caller '{caller_id[:30]}...' → LangGraph thread: {langgraph_thread_id}")
    else:
        langgraph_thread_id = thread_id_cache[cache_key]
        print(f"♻️  Reusing {environment} thread for caller '{caller_id[:30]}...' → LangGraph thread: {langgraph_thread_id}")
    
    return langgraph_thread_id


# ============================================================================
# Alternative Thread Management (Time-Based Expiry) - Currently Unused
# ============================================================================

def get_thread_id_with_expiry(caller_id: Optional[str] = None, environment: str = "prod", expiry_hours: int = 24) -> str:
    """Alternative: Get thread ID with time-based expiry (currently unused).
    
    This function implements time-based thread expiry by appending a date suffix
    to the thread ID. Threads automatically expire after the specified hours.
    
    Example:
        - Same caller on Monday → thread_id: "prod-abc123-2025-01-27"
        - Same caller on Tuesday → thread_id: "prod-abc123-2025-01-28"
        - Conversations reset daily (or based on expiry_hours)
    
    Args:
        caller_id: Unique caller identifier
        environment: Environment name ("dev", "staging", or "prod")
        expiry_hours: Hours until thread expires (default: 24)
    
    Returns:
        Thread ID with date suffix for automatic expiry
    """
    import datetime
    
    if not caller_id:
        caller_id = _generate_thread_id()
    
    # Hash the base caller_id
    base_thread_id = _hash_caller_id(caller_id, environment)
    
    # Calculate expiry bucket (rounds down to nearest expiry period)
    now = datetime.datetime.utcnow()
    hours_since_epoch = int(now.timestamp() / 3600)
    expiry_bucket = hours_since_epoch // expiry_hours
    
    # Append expiry bucket to thread ID
    # This ensures threads expire after expiry_hours
    thread_id_with_expiry = f"{base_thread_id}-{expiry_bucket}"
    
    print(f"🔗 {environment.capitalize()} thread (expires in {expiry_hours}h) for '{caller_id[:30]}...' → {thread_id_with_expiry}")
    
    return thread_id_with_expiry

def call_agent(thread_id: str, message: str, environment: str = "prod") -> str:
    """Call the LangGraph agent and return response
    
    Args:
        thread_id: Thread ID for conversation
        message: User message
        environment: Environment name ("dev", "staging", or "prod")
    """
    input_data = {"messages": [{"role": "user", "content": message}]}
    response_text = ""
    client = get_langgraph_client(environment)
    
    try:
        print(f"📡 Calling {environment} LangGraph: thread={thread_id}, assistant={GRAPH_NAME}")
        for chunk in client.runs.stream(
            thread_id,
            assistant_id=GRAPH_NAME,  # Use assistant_id parameter
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
            "/chat": "POST - Text chat with agent (testing only)",
            "/v1/chat/completions": "POST - OpenAI-compatible endpoint (for ElevenLabs widget)"
        },
        "note": "Voice handled by ElevenLabs widget - no Whisper/TTS endpoints"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Text chat endpoint for testing (no voice processing)"""
    try:
        print("=" * 60)
        print("💬 [Backend] /chat endpoint hit")
        
        data = request.json
        message = data.get('message', '')
        user_id = data.get('user_id', 'web-user')
        thread_hint = data.get('thread_token') or request.headers.get('X-Thread-Token')
        if not thread_hint and user_id:
            thread_hint = user_id
        
        print(f"📝 [Backend] Message: {message}")
        print(f"👤 [Backend] User ID: {user_id}")
        
        if not message:
            print("❌ [Backend] No message provided")
            return jsonify({'error': 'No message provided'}), 400
        
        # Detect environment from 'stage' header (sent by ElevenLabs)
        # Values: "dev", "staging", "prod" (default to "prod")
        environment = request.headers.get('stage', 'prod').lower()
        if environment not in ['dev', 'staging', 'prod']:
            environment = 'prod'
        
        # Get caller_id - prioritize ElevenLabs convo_id, fallback to caller_id or thread_hint
        caller_id = request.headers.get('convo_id') or request.headers.get('caller_id') or thread_hint
        
        # Get thread ID for this caller
        thread_id = get_thread_id_for_caller(caller_id, environment=environment)
        print(f"🧵 [Backend] Thread ID: {thread_id}")
        
        # Send to agent with thread_id for conversation memory
        print(f"🚀 [Backend] Sending to LangGraph agent...")
        
        # Call agent with routing
        response_text = call_agent(thread_id, message, environment=environment)
        
        print(f"Agent response: {response_text}")
        
        # Parse response for text and optional visualization
        parsed = parse_agent_response(response_text)
        
        response_data = {
            'response': parsed["text"]
        }
        
        # Include visualization if present
        if parsed["visualization"]:
            response_data['visualization'] = parsed["visualization"]
        
        response = jsonify(response_data)
        response.headers["X-Thread-Token"] = thread_id
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# OpenAI-Compatible Endpoint for ElevenLabs Conversational AI Integration
# ============================================================================

def stream_openai_response(thread_id: str, message: str, environment: str = "prod") -> Generator[str, None, None]:
    """
    Stream LangGraph agent responses in OpenAI-compatible format.
    This endpoint is used by ElevenLabs Conversational AI Custom LLM integration.
    
    Args:
        thread_id: Thread ID for conversation
        message: User message
        environment: Environment name ("dev", "staging", or "prod")
    """
    input_data = {"messages": [{"role": "user", "content": message}]}
    has_content = False
    accumulated_text = ""
    client = get_langgraph_client(environment)
    
    try:
        print(f"[OpenAI Endpoint] Calling {environment} LangGraph: thread={thread_id}")
        
        # Stream from LangGraph
        # assistant_id is the deployed graph name
        for chunk in client.runs.stream(
            thread_id,
            assistant_id=GRAPH_NAME,  # Use assistant_id parameter
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
        
        # Determine environment from 'stage' header (sent by ElevenLabs agents)
        # Values: "dev", "staging", "prod" (default to "prod")
        environment = request.headers.get('stage', 'prod').lower()
        if environment not in ['dev', 'staging', 'prod']:
            environment = 'prod'
        
        # Get caller_id from ElevenLabs (unique per conversation)
        # Priority: convo_id (ElevenLabs conversation ID) > caller_id > X-Thread-Token > user field
        # This ensures each conversation gets its own isolated thread
        caller_id = (
            request.headers.get('convo_id') or 
            request.headers.get('caller_id') or 
            request.headers.get('X-Thread-Token') or
            data.get('user', 'elevenlabs-user-fallback')
        )
        
        # Get thread ID for this specific caller (LangGraph manages persistence)
        thread_id = get_thread_id_for_caller(caller_id, environment=environment)
        print(f"[OpenAI Endpoint] Environment: {environment}, Caller ID: {caller_id[:30]}..., Thread ID: {thread_id}, Message: {user_message[:50]}...")
        
        # Stream response in OpenAI format
        response = Response(
            stream_with_context(stream_openai_response(thread_id, user_message, environment=environment)),
            mimetype='text/event-stream'
        )
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['X-Thread-Token'] = thread_id
        return response
        
    except Exception as e:
        print(f"[OpenAI Endpoint] Exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Helper function to parse agent responses
def parse_agent_response(response_text: str) -> dict:
    """
    Parse agent response for both text thoughts and optional visualization JSON.
    
    Format: "Text thoughts here. VISUALIZATION: {json}"
    
    Returns:
        {
            "text": "Text thoughts",
            "visualization": {...} or None
        }
    """
    import json
    import re
    
    # Look for VISUALIZATION: marker
    if "VISUALIZATION:" in response_text:
        # Split on VISUALIZATION:
        parts = response_text.split("VISUALIZATION:")
        text_part = parts[0].strip()
        
        # Extract JSON from the visualization part
        json_part = parts[1].strip()
        try:
            # Find JSON object
            match = re.search(r'\{.*\}', json_part, re.DOTALL)
            if match:
                viz_data = json.loads(match.group(0))
                return {
                    "text": text_part,
                    "visualization": viz_data
                }
        except json.JSONDecodeError as e:
            print(f"[Parse] JSON decode error: {e}")
    
    # No visualization, just return text
    return {
        "text": response_text,
        "visualization": None
    }


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    print(f"Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

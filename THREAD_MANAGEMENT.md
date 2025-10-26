# Thread Management - Caller ID to Thread Mapping

## Overview

The Flask server now implements **proper thread isolation** for each caller using deterministic hashing. This ensures each unique caller gets their own conversation thread without server-side caching.

## How It Works

### 1. Caller ID from ElevenLabs

ElevenLabs sends a unique `caller_id` header with each request:
```
caller_id: unique-phone-call-identifier-12345
```

### 2. Deterministic Hashing

The server converts `caller_id` to a deterministic thread ID:

```python
def _hash_caller_id(caller_id: str, is_staging: bool = False) -> str:
    env_prefix = "staging" if is_staging else "prod"
    combined = f"{env_prefix}:{caller_id}"
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    return f"{env_prefix}-{hash_obj.hexdigest()[:16]}"
```

**Example:**
- Caller ID: `elevenlabs-call-abc123`
- Staging: `false`
- Result: `prod-a1b2c3d4e5f6g7h8`

### 3. No Server-Side Caching

**Before (❌ Problem):**
```python
thread_cache = {
    "caller-1": "thread-id-1",
    "caller-2": "thread-id-2",
    # ... grows forever until server restart
}
```

**After (✅ Solution):**
```python
# No cache needed!
thread_id = _hash_caller_id(caller_id, is_staging)
# LangGraph manages thread persistence internally
```

## Benefits

### ✅ No Memory Leaks
- No server-side cache that grows forever
- Memory usage stays constant

### ✅ Survives Restarts
- Thread IDs are deterministic (same caller_id → same thread_id)
- Works across server restarts and multiple Railway instances

### ✅ Isolated Conversations
- Each unique caller gets their own thread
- No context collision between callers

### ✅ Environment Separation
- Production threads: `prod-{hash}`
- Staging threads: `staging-{hash}`
- Same caller_id maps to different threads in prod vs staging

## Thread Lifecycle

### When Does a Thread Get Created?

LangGraph creates a thread the **first time** it receives a thread_id:

```python
# First call with this thread_id
thread_id = "prod-a1b2c3d4e5f6g7h8"
client.runs.stream(thread_id, ...)  # LangGraph creates thread

# Subsequent calls with same thread_id
client.runs.stream(thread_id, ...)  # LangGraph retrieves existing thread
```

### When Does Conversation History Persist?

- ✅ **Same caller_id** = Same thread = Conversation continues
- ✅ **Across server restarts** = Thread persists (stored in LangGraph)
- ✅ **Multiple Railway instances** = Thread accessible from all instances

### When Does a Thread Expire?

**LangGraph manages expiration** based on its own policies:
- Typically threads expire after **X days of inactivity**
- Exact policy depends on LangGraph Cloud configuration
- No manual cleanup needed on our side

## Example Flow

### Scenario: Two Callers Call the Agent

**Caller A (Phone: +1-555-0001):**
```
1. ElevenLabs sends: caller_id = "elevenlabs-call-001"
2. Server hashes: thread_id = "prod-abc123def456"
3. LangGraph creates thread "prod-abc123def456"
4. Caller A: "What recipes do you have?"
5. Agent responds with recipes
```

**Caller B (Phone: +1-555-0002):**
```
1. ElevenLabs sends: caller_id = "elevenlabs-call-002"
2. Server hashes: thread_id = "prod-xyz789ghi012"
3. LangGraph creates thread "prod-xyz789ghi012"
4. Caller B: "What recipes do you have?"
5. Agent responds with recipes (independent conversation)
```

**Caller A Calls Again:**
```
1. ElevenLabs sends: caller_id = "elevenlabs-call-001" (same as before)
2. Server hashes: thread_id = "prod-abc123def456" (same as before)
3. LangGraph retrieves existing thread
4. Conversation continues from where it left off!
```

## Implementation Details

### Function: `get_thread_id_for_caller()`

```python
def get_thread_id_for_caller(caller_id: Optional[str] = None, is_staging: bool = False) -> str:
    """Get deterministic thread ID for a caller.
    
    No caching needed - LangGraph manages thread persistence internally.
    Each unique caller_id maps to the same thread_id consistently.
    """
    if not caller_id:
        caller_id = _generate_thread_id()  # Random fallback
    
    thread_id = _hash_caller_id(caller_id, is_staging)
    return thread_id
```

### Endpoints Updated

All endpoints now use the new function:

1. **`/v1/chat/completions`** (OpenAI-compatible for ElevenLabs)
2. **`/chat`** (Text chat with TTS)
3. **`/process_voice`** (Voice with TTS)
4. **`/process_voice_text_only`** (Voice to text)

### Caller ID Priority

The server checks for `caller_id` in this order:

1. `caller_id` header (from ElevenLabs)
2. `X-Thread-Token` header (custom)
3. `user` field in request body (fallback)
4. Random UUID (last resort)

## Logging

Server logs show thread mapping:

```
🔗 Production thread for caller 'elevenlabs-call-abc123...' → thread_id: prod-a1b2c3d4e5f6g7h8
```

Or if no caller_id:

```
⚠️  No caller_id provided, generated random thread: 550e8400-e29b-41d4-a716-446655440000
```

## Testing

### Test with curl

```bash
# Caller 1
curl -X POST https://your-server.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "caller_id: test-caller-001" \
  -H "staging: false" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "agent"
  }'

# Caller 2 (different thread)
curl -X POST https://your-server.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "caller_id: test-caller-002" \
  -H "staging: false" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "agent"
  }'

# Caller 1 again (same thread, conversation continues)
curl -X POST https://your-server.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "caller_id: test-caller-001" \
  -H "staging: false" \
  -d '{
    "messages": [{"role": "user", "content": "What did I just ask?"}],
    "model": "agent"
  }'
```

## Migration Notes

### What Changed

- ❌ Removed: `thread_cache` dictionary
- ❌ Removed: `get_or_create_thread()` function
- ✅ Added: `get_thread_id_for_caller()` function
- ✅ Added: `_hash_caller_id()` for deterministic hashing

### No Breaking Changes

- All endpoints still work the same way
- `X-Thread-Token` header still returned in responses
- Conversation continuity maintained (better than before!)

---

**Built for foodweb.ai** 🍽️

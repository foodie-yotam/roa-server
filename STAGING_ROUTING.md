# Staging/Production Routing

## Overview

The Flask server now supports routing requests to **both staging and production** LangGraph deployments based on the `staging` header sent by ElevenLabs agents.

## How It Works

### ElevenLabs Agent Configuration

Both ElevenLabs agents send a custom header to identify their environment:

- **ROA-prod**: Sends `staging: false`
- **ROA-staging**: Sends `staging: true`

### Server Routing Logic

The server detects the `staging` header and routes requests accordingly:

```python
is_staging = request.headers.get('staging', 'false').lower() == 'true'
```

- If `staging: true` → Routes to **staging LangGraph deployment**
- If `staging: false` or header missing → Routes to **production LangGraph deployment**

### Separate Thread Management

Conversations are kept separate between environments:
- Production threads: `{user_id}:prod`
- Staging threads: `{user_id}:staging`

This ensures that staging experiments don't interfere with production conversations.

## Deployments

### Production
- **URL**: `https://roa-voice-prod-59235014c2ad5a5f830e9e124171824f.us.langgraph.app`
- **Database**: Production Neo4j (`e3726068.databases.neo4j.io`)
- **Branch**: `main`

### Staging
- **URL**: `https://roa-voice-staging-66a6f6ec95d9546995a4f3352ad05df2.us.langgraph.app`
- **Database**: Staging Neo4j (`98c4d351.databases.neo4j.io`)
- **Branch**: `staging`

## Environment Variables

Update Railway with these environment variables:

```bash
# Production LangGraph
LANGGRAPH_URL_PROD=https://roa-voice-prod-59235014c2ad5a5f830e9e124171824f.us.langgraph.app

# Staging LangGraph
LANGGRAPH_URL_STAGING=https://roa-voice-staging-66a6f6ec95d9546995a4f3352ad05df2.us.langgraph.app

# LangSmith API Key (same for both)
LANGSMITH_API_KEY=your_langsmith_api_key

# Other existing vars
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
```

## Affected Endpoints

All endpoints now support staging/production routing:

1. **`/v1/chat/completions`** (OpenAI-compatible for ElevenLabs)
   - Detects `staging` header
   - Routes to appropriate LangGraph deployment
   - Maintains separate conversation threads

2. **`/chat`** (Text chat with TTS)
   - Detects `staging` header
   - Routes to appropriate deployment

3. **`/process_voice_text_only`** (Voice to text)
   - Detects `staging` header
   - Routes to appropriate deployment

## Testing

### Test Production Routing
```bash
curl -X POST https://your-railway-app.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "staging: false" \
  -d '{
    "messages": [{"role": "user", "content": "List all kitchens"}],
    "model": "agent",
    "stream": true
  }'
```

### Test Staging Routing
```bash
curl -X POST https://your-railway-app.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "staging: true" \
  -d '{
    "messages": [{"role": "user", "content": "Create a kitchen called Test Kitchen"}],
    "model": "agent",
    "stream": true
  }'
```

## Logs

Server logs now indicate which environment is being used:

```
[OpenAI Endpoint] Environment: production, User: user123, Thread ID: abc...
[OpenAI Endpoint] Calling production LangGraph: thread=abc...
```

```
[OpenAI Endpoint] Environment: staging, User: user456, Thread ID: def...
[OpenAI Endpoint] Calling staging LangGraph: thread=def...
```

## Benefits

✅ **Single Server**: One Flask deployment handles both environments  
✅ **Automatic Routing**: No manual configuration needed  
✅ **Separate Threads**: Staging and production conversations don't mix  
✅ **Easy Testing**: CEO can experiment on staging without affecting production  
✅ **Transparent**: Logs clearly show which environment is being used  

---

**Built for foodweb.ai** 🍽️

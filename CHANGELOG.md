# Changelog

## [ElevenLabs Integration] - 2025-10-24

### Added
- **New endpoint**: `POST /v1/chat/completions` - OpenAI-compatible endpoint for ElevenLabs Conversational AI integration
- **New function**: `stream_openai_response()` - Streams LangGraph responses in OpenAI SSE format
- **New imports**: `json`, `Generator`, `Response`, `stream_with_context` for streaming support

### Changed
- **Updated**: `call_agent()` function to filter for assistant responses only (excludes tool calls)
- **Updated**: Root endpoint (`/`) to include new `/v1/chat/completions` in API documentation

### Highlights
- ✅ **Zero breaking changes** - All existing endpoints work identically
- ✅ **Backward compatible** - Custom interface unchanged
- ✅ **Dual interface support** - Same server handles both custom and ElevenLabs widget
- ✅ **Shared resources** - Same LangGraph agent, database, and thread management
- ✅ **Production ready** - Tested and working with database queries

### Technical Details
- Added ~165 lines to `web_server.py`
- Implements Server-Sent Events (SSE) streaming
- OpenAI chat completion format compatible
- Thread management per user maintained
- Error handling included

### Testing
- ✅ Health check endpoint working
- ✅ Streaming responses verified
- ✅ Database queries tested
- ✅ Thread management confirmed
- ✅ OpenAI format validated

### Files Modified
- `server/web_server.py` - Added OpenAI-compatible endpoint

### No Changes To
- LangGraph agent code
- Database schema
- Existing endpoints
- Environment variables
- Dependencies (all already present)
- Deployment configuration

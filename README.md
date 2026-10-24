# Origami

Context distillation and memory management service using SystemOne.

Origami dissects conversation segments, grades them for truthfulness and usefulness,
banks stale/redundant segments outside the active context, and produces distilled
context sets for the next turn.

## Features

- **Segment dissection** — breaks conversations into atomic segments
- **SystemOne grading** — truthfulness and usefulness scores via SystemOne API
- **Banking system** — parks stale segments outside active context for later recall
- **Anti-fact flagging** — marks segments as CONTESTED (gray zone) or FALSE (confirmed wrong)
- **Task-specific instructions** — parks instructions until their task is active
- **Separate endpoints** — supports different SystemOne (decision) vs chat/completion endpoints

## Installation

```bash
pip install -e .
```

## Usage

```python
from origami import ContextManager, OrigamiConfig

# Configure separate SystemOne and chat endpoints
config = OrigamiConfig(
    systemone_base_url="http://127.0.0.1:8005",
    systemone_endpoint="/v1/systemone",
    systemone_api_key="token-abc123",
    chat_base_url="http://127.0.0.1:8080",
    chat_endpoint="/v1/chat/completions",
    chat_api_key="sk-...",
    model="Qwopus3.8-27B-Flash-Q4_K_M"
)

# Initialize manager
manager = ContextManager(config=config)

# Add conversation data
manager.add(
    messages=[{"role": "user", "content": "Hello"}],
    state="Discussing project goals"
)

# Distill context
result = manager.distill(max_tokens=4096)
print(f"Kept: {len(result['kept'])}, Banked: {len(result['banked'])}")

# Get distilled context
segments = manager.get_context(max_tokens=4096)

# Recall banked segments when relevant
recalled = manager.recall_bank(state="Discussing project timeline")

# Fact-check a claim
check = manager.fact_check("The project deadline is Friday")
print(f"Verdict: {check['verdict']}, Score: {check['score']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ORIGAMI_SYSTEMONE_URL` | SystemOne base URL | `http://127.0.0.1:8005` |
| `ORIGAMI_SYSTEMONE_API` | SystemOne API key | `token-abc123` |
| `ORIGAMI_CHAT_URL` | Chat/completions base URL | `http://127.0.0.1:8005` |
| `ORIGAMI_CHAT_API` | Chat/completions API key | `token-abc123` |
| `ORIGAMI_MODEL` | Model name | `Qwopus3.8-27B-Flash-Q4_K_M` |
| `ORIGAMI_STORAGE` | Storage path (None for in-memory) | `None` |

### Separate Endpoints

Origami supports routing SystemOne (decision/scoring) calls to a different endpoint
than chat/completion (LLM) calls:

```python
config = OrigamiConfig(
    # SystemOne calls go here
    systemone_base_url="http://127.0.0.1:8005",
    systemone_endpoint="/v1/systemone",
    systemone_api_key="token-systemone",
    
    # Chat calls go here
    chat_base_url="http://127.0.0.1:8080",
    chat_endpoint="/v1/chat/completions",
    chat_api_key="sk-chat-..."
)
```

## Anti-Fact Flagging

Origami uses a gray-scale truthfulness system:

- **ACTIVE** — verified true, high priority
- **CONTESTED** — disputed/unverified, reduced priority (50%)
- **FALSE** — confirmed wrong, priority = 0, excluded from active context

```python
# Mark a segment as contested
manager.mark_contested(
    seg_id="seg-123",
    reason="Conflicts with verified data",
    confidence=0.7,
    evidence=["Source A says X", "Source B says Y"]
)

# Mark a segment as false
manager.mark_false(
    seg_id="seg-456",
    reason="Factually incorrect",
    corrected_content="Corrected version",
    evidence=["Official documentation"]
)

# Audit all facts
audit = manager.audit_facts(state="Current project state")
print(f"Verified: {audit['verified']}, Contested: {audit['contested']}, False: {audit['false']}")
```

## Task-Specific Banking

Park instructions for specific tasks and recall them when the task is active:

```python
# Bank an instruction for a specific task
manager.add_instruction(
    content="Use 2-space indentation for all code",
    task_id="code-formatting",
    tags=["code", "style"]
)

# Later, recall when task is active
manager.recall_by_task("code-formatting")
```

## License

[MIT](LICENSE)

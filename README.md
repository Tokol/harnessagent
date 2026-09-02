# Harness Agent POC

A Streamlit proof of concept for a governed agent with an entity map, bounded JSON state, an action registry, policies, snapshots, and segmented archives.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Create `.env` from `.env.example` and set a real OpenAI API key. The app uses the Responses API with function tools; there is no mock or regex intent parser.

- `Remember that Falano uses PostgreSQL`
- `Create a task to review Falano`
- `Recall Falano`
- `Update that memory to say Falano uses SQLite instead`
- `Forget that memory`
- `Show my active tasks`
- `Create a snapshot`

Runtime data is created under `agent_data/` and intentionally ignored by Git. Live entity files retain 20 records; overflow is moved into dated archive files of up to 100 records each. Writes use atomic file replacement.

The UI is intentionally a plain conversation. Internally, the latest 20 chat messages are kept as active context. Older messages rotate into 100-record dated history archives. The agent can search current and archived history, memory, tasks, skills, and reflections through its governed `context_search` tool.

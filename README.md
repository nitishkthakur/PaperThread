# PaperThread

**What research paper should I read next, and why?**

PaperThread builds a reading path through research literature. Not a search engine and not a
similarity recommender — it models a reader's journey through a field and works out which paper
does the most for them next: the missing prerequisite, the foundational ancestor they skipped, the
logical next step.

Every recommendation comes with an explanation: why the paper matters, what it assumes, what it
teaches, and why it's the right one for this reader.

## Status

Working, and honest about where it isn't. Enter a topic and you get an ordered reading path:
what to read first, the paper that *is* the topic, and where the idea went afterwards.

Ask for **dropout in neural networks** and it builds:

> backpropagation → weight decay → bagging → **Dropout (Srivastava et al.)** → analyses of why
> it works → variational and concrete dropout

None of the first three papers mention dropout. They are there because you cannot evaluate
dropout's central claim — that it approximates an ensemble — without them.

**Every path states how much it trusts itself**, because the system is sometimes wrong and a
wrong path reads exactly like a right one otherwise. Papers an LLM names are looked up against
real providers and dropped if they cannot be found, never quietly substituted.

Not built: reading history, marking papers read, and personalization. Until those exist the path
is built for the topic rather than for you, and the interface says so.

## Running it

Requires Python 3.11+ and Node 18+.

```bash
# backend
cd backend
python3.11 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m uvicorn paperthread.api.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

**For LLM-planned paths** (the good ones), run `ollama signin` — the local daemon then reaches
Ollama Cloud models with no API key. Without it the system falls back to a citation-graph-only
pipeline and reports that it did. Setting `OPENALEX_API_KEY` (free) is also worth it: OpenAlex is
metered now, and the anonymous tier is about a hundred searches a day.

## Design

- **Everything is multi-provider.** Paper sources and LLM providers are selected in one config
  file (`config/paperthread.toml`); adding another means writing an adapter, not changing calling
  code.
- **Retrieval is layered, and the bottom layer works offline.** Lexical and citation-graph
  analysis need no model weights and no network beyond the paper sources themselves. Embeddings,
  reranking, and LLM reasoning refine that result; disabling any of them degrades quality without
  breaking the system.
- **Local-first, but a web app from day one** — built to become multi-user rather than rewritten
  into it.
- **The ranking algorithm is configuration, not code.** Which strategy builds a path, which model
  plans it, and the prompts it uses are all set in one file and measured with a repeatable
  evaluation harness (`backend/tools/`). Citation centrality finds the *important* papers; it does
  not find a *teaching order*, and knowing which approach wins required measuring rather than
  arguing.

## Documentation

| Document | Contents |
|---|---|
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | What the product must do, and the decisions behind it |
| [`docs/PROVIDER_NOTES.md`](docs/PROVIDER_NOTES.md) | How external providers differ, and what that forces |
| [`docs/RETRIEVAL_NOTES.md`](docs/RETRIEVAL_NOTES.md) | Evidence behind the retrieval design |
| [`AGENTS.md`](AGENTS.md) | Repository map and conventions |

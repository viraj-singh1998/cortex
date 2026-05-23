# ML Engineer prep plan — Viraj Singh
> Reading + implementation, integrated. 5 weeks, 2–3 hrs/day. Goal: interviews + field mastery.

---

## How to use this plan

Each phase has **study modules** (reading, concepts, interview Q&A) and **a build** that runs in parallel. The build isn't a separate track — it's how you consolidate what you're reading. Timings assume 2–3 hrs/day: roughly 1–1.5 hrs on reading/concepts, 1–1.5 hrs on the build.

**Colour code used below:**
- 🟢 Your direct experience (Skyfall / FNFI) — go deep, be ready to defend
- 🔵 Field knowledge — need fluency, not mastery
- 🔴 Your flagged gap — needs a clear POV

---

## Phase 1 — Foundation & your strongest ground (days 1–10)

> Ship early. Start from what you've already built. The first two builds are grounded in your Skyfall work — lower activation energy, faster to get something live on GitHub.

---

### Module 1 🟢 — Memory systems

**Your work:** Tastes/Tasty — preference capture, short + long-term memory, contradiction resolution, confidence scoring, Ebbinghaus decay, isolation pattern. Informed by MemGPT, mem0, LangMem, PAMU, Graphiti/Zep.

#### Concepts to know cold

**The taxonomy**
| Type | What it stores | Example |
|---|---|---|
| Episodic | Past events, conversation history | "Last week you asked about fintech VCs" |
| Semantic | Facts, preferences, beliefs | "User prefers bullet-point summaries" |
| Procedural | Workflows, how-tos | "User's PR process: draft → review → merge" |
| Working | Current session state | Active context window |

**System architectures**
- **MemGPT / Letta** (`arXiv:2310.08560`): two-tier memory — core memory in-context + archival memory external. Self-editing via tools. OS analogy: LLM = CPU, context window = RAM, archival = disk.
- **PAMU** (`arXiv:2510.09720`): sliding window average (SW) + exponential moving average (EMA) fused. Captures short-term fluctuations and long-term tendencies. Detects preference shifts.
- **Graphiti/Zep** (`arXiv:2501.13956`): bi-temporal knowledge graph. Every fact has two timestamps — event time (when it was true in the world) and ingestion time (when the agent learned it). Enables temporal reasoning.
- **mem0** (`arXiv:2504.19413`): multi-signal retrieval — semantic + BM25 + entity linking. Entity extraction during `add()`, entity matching during `search()`. Boosts precision significantly. New token-efficient algorithm (2026): +29.6 pts on temporal queries, +23.1 pts on multi-hop, under 7K tokens/retrieval vs 25K+ for full-context.
- **LangMem**: episodic + semantic + procedural memory via LangGraph's BaseStore. Procedural memory = agents updating their own system prompts. Note: p95 latency ~59s on LOCOMO — not for interactive agents.
- **A-MEM** (`arXiv:2502.12110`, Feb 2025): Zettelkasten-style agentic memory. On `add()`, generates a structured note (contextual description + keywords + tags), then scans for related memories and links them. As new memories arrive, they retroactively update the attributes of existing ones — the network continuously refines itself. SOTA on 6 foundation models.
- **Memori** (`arXiv:2603.19935`, Mar 2026): converts dialogue into compact semantic triples + summaries (no raw text stored). Achieves 81.95% on LoCoMo using only 1,294 tokens/query — 67% fewer tokens than competing approaches and 20× savings vs full-context. LLM-agnostic.
- **Memoria** (`arXiv:2512.12686`, Dec 2025): weighted KG-based user modelling engine + session-level summarisation. Incremental entity/preference capture. 38.7% latency reduction and lower token usage on LongMemEval vs prior SOTA.
- **Honcho** (product, 2026): dialectic user modelling — a 3-agent pipeline (Honcho 3) builds a persistent "dialect" of each user's preferences *implicitly*, without requiring the user to state them. Ingestion Reasoning captures explicit signals in parallel. Strongest product for per-user preference memory.

**Your Tasty pattern:** isolating memory ops into a separate CLI thread makes preference capture a hard requirement rather than an instruction the agent can skip. This is the "Isolate" strategy from the context engineering framework. Know it cold.

**Confidence scoring:** every memory has a score that decays over time (Ebbinghaus forgetting curve) and strengthens with reinforcement. Contradictions reduce confidence; consistent reinforcement increases it.

**Contradiction resolution strategies:** last-write-wins (simple, lossy), confidence-weighted (keep higher-confidence fact), ask-user (best quality, worst UX for background tasks), version both (expensive but lossless).

**Benchmarks to know:**
- **LOCOMO** — 300+ turn conversations, the standard for preference/episodic memory. Memori hits 81.95% here.
- **LongMemEval** — long-session memory across multiple turns. Memoria benchmarks against this.
- **MemBench** — comprehensive agent memory evaluation (ACL 2025).
- **PERMA** (`arXiv:2603.23231`) — personalisation and memory benchmark.
- **BEAM** (Beyond a Million Tokens, 2026) — the hardest benchmark. 100 conversations × up to 10M tokens, 2,000 probing questions across 10 categories. Key finding: preference following scores 88% but contradiction resolution scores only 36% — the hardest open problem. Two tracks: BEAM-1M and BEAM-10M. Directly relevant to cortex-memory's contradiction resolution module.
- **MemoryAgentBench** (`arXiv:2507.05257`, ICLR 2026) — evaluates 4 competencies: accurate retrieval, test-time learning, long-range understanding, and **selective forgetting**. Repurposes existing long-context datasets into incremental multi-turn format + introduces EventQA and FactConsolidation datasets.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| MemGPT — Packer et al., 2023 | `arXiv:2310.08560` |
| PAMU — preference-aware memory | `arXiv:2510.09720` |
| Graphiti/Zep — bi-temporal KG | `arXiv:2501.13956` |
| mem0 — multi-signal memory | `arXiv:2504.19413` |
| Anatomy of agentic memory | `arXiv:2602.19320` |
| PERMA benchmark | `arXiv:2603.23231` |
| A-MEM — Zettelkasten agentic memory | `arXiv:2502.12110` |
| Memori — semantic triples + persistent layer | `arXiv:2603.19935` |
| Memoria — KG user modelling + session summary | `arXiv:2512.12686` |
| Memory for Autonomous LLM Agents (survey) | `arXiv:2603.07670` |
| From Experience to Strategy: Graph Memory | `arXiv:2511.07800` |
| MemoryAgentBench (ICLR 2026) | `arXiv:2507.05257` |

#### Products to know

| Product | What it does | Standout |
|---|---|---|
| **mem0** | Managed memory layer — vector + graph + KV store, auto extraction | Largest community (47K+ GitHub stars), 21 framework integrations |
| **Zep** | Temporal KG via Graphiti engine, episodic sequences | Best for temporal reasoning across sessions |
| **Letta (MemGPT)** | Stateful memory runtime, editable memory blocks | Memory as first-class agent state |
| **LangMem** | Long-term memory via LangGraph BaseStore | Tight LangGraph integration, procedural memory (agents update own system prompt) |
| **Honcho** | Dialectic user modelling, implicit preference capture | Only product purpose-built for per-user preference memory |
| **Supermemory** | Context fencing, session graph, hybrid retrieval | Full-platform play, minimal assembly required |
| **Hindsight** | KG-based recall + cross-memory `reflect` synthesis tool | Cross-memory synthesis is unique; fewest deps |
| **MemPalace** | Verbatim recall, personal archive | Best for exact-fact retrieval at scale |

#### Blogs & articles

| Resource | Link |
|---|---|
| State of AI Agent Memory 2026 (mem0) | mem0.ai/blog/state-of-ai-agent-memory-2026 |
| AI Memory Benchmarks in 2026 (mem0) | mem0.ai/blog/ai-memory-benchmarks-in-2026 |
| Benchmarked: OpenAI Memory vs LangMem vs MemGPT vs Mem0 | mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0 |
| Token-Efficient Memory Algorithm (mem0 research) | mem0.ai/research |
| Anthropic: Effective harnesses for long-running agents | anthropic.com/engineering/effective-harnesses-for-long-running-agents |
| LangChain: Your harness, your memory | langchain.com/blog/your-harness-your-memory |
| The New Stack: Memory — a new paradigm of context engineering | thenewstack.io/memory-for-ai-agents-a-new-paradigm-of-context-engineering |
| Context Engineering in 2025 (mem0) | mem0.ai/blog/context-engineering-ai-agents-guide |
| Practical Guide to Memory for Autonomous LLM Agents (TDS) | towardsdatascience.com/a-practical-guide-to-memory-for-autonomous-llm-agents |
| Agent Memory Providers Compared (Honcho, Mem0, Hindsight…) | glukhov.org/ai-systems/memory/agent-memory-providers |

#### Interview questions
1. Walk me through how Tasty isolates memory ops from the main agent thread. Why does that matter for token efficiency?
2. How did you handle contradictions between old and new preferences in Tastes?
3. Compare MemGPT's two-tier architecture to mem0's approach. When would you pick one over the other?
4. How would you evaluate a memory system? What metrics would you design for Tastes specifically?
5. What's a bi-temporal knowledge graph and why does it matter for agent memory?
6. BEAM shows contradiction resolution scores ~36% while preference following scores ~88%. Why is contradiction resolution so much harder, and how would you improve it?
7. What is A-MEM's Zettelkasten approach and when does interconnected memory beat flat storage?
8. Memori achieves 81.95% on LoCoMo with 1,294 tokens/query. What architectural choice makes that possible and what does it sacrifice?
9. How does Honcho capture preferences implicitly without the user stating them? What's the failure mode?
10. What's "selective forgetting" (MemoryAgentBench) and why does an agent need it?

---

### Module 2 🟢 — Agent architecture & orchestration

**Your work:** Multi-agent systems across enterprise IT agent and workplace copilot. ReAct, CoT, Plan-and-Act, parent-child orchestration. Direct APIs + custom and OSS MCP servers.

#### Concepts to know cold

- **ReAct loop** (`arXiv:2210.03629`): think → act → observe → repeat. Reasoning and acting are interleaved so the agent adapts mid-task. Fails when chains get long (hallucination, error cascades).
- **Plan-and-Solve / Plan-and-Execute** (`arXiv:2305.04091`): planner generates full plan upfront, executor runs steps. More deterministic, less adaptive. Good when the task is well-scoped.
- **Parent-child orchestration:** high-level planner decomposes tasks and delegates to specialised subagents. Key design questions: how do you pass context? How do you handle child failures? How do you prevent the parent from micromanaging?
- **CoT vs ReAct vs Tree-of-Thoughts:** CoT = reasoning traces only; ReAct = reasoning + tool calls; ToT = explores multiple reasoning branches (expensive, rarely used in production).
- **When to use which:** ReAct for adaptive, tool-heavy tasks. Plan-and-Execute for deterministic multi-step workflows. Parent-child for tasks requiring genuine specialisation.
- **MCP:** tools vs resources vs prompts. Tools = callable functions. Resources = read-only data (files, DB records). Prompts = reusable templates. Know when to build a custom MCP server vs use an OSS one.
- **Failure handling:** circuit breakers, retry strategies, loop detection, fallback to human-in-the-loop.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| ReAct — Yao et al., 2022 | `arXiv:2210.03629` |
| Plan-and-Solve prompting | `arXiv:2305.04091` |
| CoALA — cognitive architectures for agents | `arXiv:2309.02427` |
| Survey on LLM-based autonomous agents | `arXiv:2308.11432` |
| Anthropic — building effective agents | anthropic.com/research/building-effective-agents |
| LangGraph multi-agent docs | langchain-ai.github.io/langgraph |
| MCP specification | spec.modelcontextprotocol.io |

#### Interview questions
1. When do you pick ReAct vs Plan-and-Execute? Give a concrete example from your work.
2. How did you prevent your orchestrator from micromanaging subagents or looping?
3. What's the tradeoff between direct API integrations and MCP servers?
4. How do you pass context between a parent agent and a child agent safely?
5. What failure modes did you hit with long-horizon tasks and how did you mitigate them?

---

### Module 3 🟢 — Code generation as tool calling

**Your work:** Replaced traditional tool-calling with a code-generation agent querying Postgres directly via SDK. Reduced complex query execution from 2–5 min to under 60 seconds.

#### Concepts to know cold

- **Why code-gen beats tool-calling at scale:** tool schemas are rigid and require maintenance per integration. A code-gen agent that understands an SDK can generalise to any query the SDK supports — document the SDK once, get all queries for free.
- **The tradeoff:** code-gen is harder to sandbox, harder to debug, requires SDK docs to be well-indexed in context. Tool-calling is safer and more predictable for simple, well-defined actions.
- **Execution sandboxing:** Docker containers, restricted environments (no filesystem/network access by default), resource limits, timeout enforcement.
- **SDK-as-context:** indexing a Python SDK into a context-friendly doc the LLM can query at runtime. Chunking the SDK by function, providing type signatures and docstrings.
- **The spectrum:** pure tool-calling → code-gen for data queries → code-gen for workflow construction → fully autonomous code-gen.
- **PAL (Program-Aided Language Models):** the foundational paper for code-as-reasoning. The LLM writes a program to solve a problem and delegates execution to an interpreter.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| PAL — Program-Aided Language Models | `arXiv:2211.10435` |
| ToolBench — tool learning survey | `arXiv:2304.08354` |
| LangChain — filesystems for context | blog.langchain.com |

#### Interview questions
1. Why did you replace tool-calling with code generation? What were the risks you accepted?
2. How did you provide SDK context to the LLM without blowing the context window?
3. How did you sandbox generated code? What could go wrong without sandboxing?
4. When would you revert to traditional tool-calling over code-gen?

---

### Build 1 🟢 — Tasty: open-source memory library (days 3–10)

**What:** Extract, clean, and publish your CLI preference engine as a reusable, installable Python package.

**Why:** You already built this. Packaging it as OSS forces you to articulate design decisions, write docs, and think about the API surface — all of which are interview gold. Gives you a real GitHub project with commit history that maps directly to your resume bullet.

**Stack:** Python, LangGraph, Anthropic API, PyPI packaging, pytest

**Milestones:**
1. Refactor Tasty into a clean library structure with a public API (`add`, `resolve`, `inject`)
2. Write a contradiction resolution module with configurable strategy (last-wins, confidence-weighted, ask-user)
3. Add confidence scoring with Ebbinghaus decay — time-aware memory scoring
4. Write a LOCOMO-style eval harness to benchmark preference capture rate
5. Publish to PyPI, write a detailed README with architecture diagram
6. Blog: *"How I built a preference memory engine for LLM agents — and why I isolated it from the main thread"*

**Deliverables:**
- GitHub repo — installable Python package with tests and CI
- README with architecture, design decisions, and usage examples
- Eval notebook: preference capture rate, contradiction resolution accuracy
- Blog post: memory isolation pattern + Ebbinghaus decay implementation

---

## Phase 2 — RAG, inference & the systems layer (days 11–24)

> Fill the gaps below your strongest ground. RAG and inference optimisation are areas where your resume has work but the depth story needs sharpening. LLM internals and context engineering are field knowledge that every senior ML engineer needs fluency in.

---

### Module 4 🟢 — RAG & retrieval engineering

**Your work:** Workplace copilot RAG pipeline — structured filesystem + Qdrant vector store for context retrieval.

#### Concepts to know cold

- **Hybrid retrieval:** dense (vector/semantic) + sparse (BM25/keyword). Dense is good for semantic similarity; sparse is good for exact matches and rare terms. Reciprocal Rank Fusion (RRF) is the standard combination method.
- **Chunking strategies:** fixed-size, sentence-based, paragraph-based, semantic chunking. Chunk size affects retrieval quality significantly — too small loses context, too large retrieves noise.
- **Cross-encoder re-ranking:** after first-stage retrieval, use a cross-encoder (Cohere Rerank, BGE-reranker) to re-score top-k results. Significant quality boost for minimal latency cost.
- **Context compression:** LLMLingua, RECOMP — compress retrieved chunks before injecting into context to save tokens without losing information.
- **Corrective RAG (CRAG):** self-correcting retrieval that falls back to web search when internal retrieval confidence is low.
- **HyDE (Hypothetical Document Embeddings):** generate a hypothetical answer first, embed it, use that embedding for retrieval. Improves recall for complex queries.
- **Filesystem as context store:** agents navigate a well-indexed filesystem (with an index file the agent reads first) rather than stuffing everything in a vector DB. Scales differently — better for structured org knowledge.
- **Qdrant specifics:** payload filtering, sparse vector support, named vectors, collection configuration.
- **RAGAS metrics:** faithfulness (does the answer follow from the retrieved context?), answer relevance (is the answer relevant to the question?), context precision (are retrieved chunks actually useful?), context recall (did we retrieve what was needed?).

#### Papers & resources

| Resource | ID / Link |
|---|---|
| RAG — Lewis et al., 2020 | `arXiv:2005.11401` |
| RAGAS — automated RAG eval | `arXiv:2309.15217` |
| Corrective RAG (CRAG) | `arXiv:2401.15884` |
| Qdrant hybrid search docs | qdrant.tech/documentation/concepts/hybrid-queries |
| LangChain context engineering blog | blog.langchain.com/context-engineering-for-agents |

#### Interview questions
1. How did you decide what to put in Qdrant vs the filesystem?
2. Walk me through your chunking strategy and why you chose it.
3. How would you evaluate retrieval quality end-to-end?
4. What's Reciprocal Rank Fusion and when does hybrid search outperform pure dense?
5. How did you handle stale or outdated documents?

---

### Module 5 🔵 — Context engineering

**The meta-skill behind every agent decision.**

#### Concepts to know cold

- **Karpathy framing:** "LLM is the CPU, context window is the RAM." Context engineering is the OS that decides what to load into RAM and when. This is a better framing than "prompt engineering" for production systems.
- **The four strategies (LangChain framework):**
  - **Write** — save context externally (filesystem, DB, vector store) so it can be retrieved later
  - **Select** — retrieve only the relevant subset of context at each step
  - **Compress** — summarise, trim, or distil context to fit within the window
  - **Isolate** — use subagents or tools to handle specific tasks with their own context, keeping the main thread clean. **This is exactly what Tasty does.**
- **Why context engineering > prompt engineering:** prompts are a tiny fraction of total context in production agents. The rest is conversation history, tool outputs, retrieved docs, agent state.
- **KV cache awareness:** prompt caching (Anthropic, OpenAI) lets you cache the static prefix of your system prompt. Design your context so the cacheable part (instructions, tools, personas) comes first and the dynamic part (user message, retrieved docs) comes last.
- **Token budget management:** at scale, every token costs money. Know when to compress, when to offload to external storage, when to truncate.
- **12-factor agents:** production agent design principles — treat agents like 12-factor apps. Stateless execution, external state stores, idempotent actions, structured outputs.
- **Context poisoning:** injecting adversarial content into retrieved context to manipulate agent behaviour. Know the attack and the mitigations.

#### Resources

| Resource | Link |
|---|---|
| LangChain context engineering blog | blog.langchain.com/context-engineering-for-agents |
| LangChain context engineering repo | github.com/langchain-ai/context_engineering |
| 12-factor agents | github.com/humanlayer/12-factor-agents |
| Karpathy on context engineering | x.com/karpathy/status/1937902205765607626 |

#### Interview questions
1. Explain the write/select/compress/isolate framework with examples from your own systems.
2. How do you decide when to compress vs evict context?
3. What is prompt caching and how does it affect agent design?
4. Where does context engineering end and RAG begin?

---

### Module 6 🟢 — LLM inference & serving

**Your work:** 20× throughput improvement at FNFI using vLLM batched inference, 60% cost reduction across ~2.5M documents/month.

#### Concepts to know cold

- **Paged Attention** (`arXiv:2309.06180`): KV cache is the bottleneck for LLM serving. Paged Attention manages the KV cache like OS virtual memory — pages allocated non-contiguously, enabling much higher batch sizes and GPU utilisation.
- **Continuous batching:** new requests join the batch as soon as a sequence finishes, no wasted GPU cycles. This is the default in vLLM and why it dramatically outperforms static batching.
- **Prefill vs decode:** prefill = processing the input prompt (parallelisable, fast). Decode = generating tokens one at a time (sequential, the bottleneck). Know this distinction — it explains why long prompts have high TTFT (time to first token) but short generation.
- **Speculative decoding:** a small draft model generates k tokens, the large model verifies them in one forward pass. If they match, you get k tokens for the cost of one verification step. Reduces decode latency.
- **Flash Attention I & II:** recomputes attention in blocks to avoid materialising the full attention matrix. Reduces memory from O(n²) to O(n) and speeds up training/inference significantly.
- **Quantisation:** INT8, INT4, GPTQ, AWQ. Trade memory and speed for slight quality loss. Know the tradeoff for your use case.
- **LoRA** (`arXiv:2106.09685`): add low-rank adapter matrices (rank r) to frozen base model weights. Train only the adapters — drastically fewer parameters. The low-rank decomposition works because weight update matrices are empirically low-rank.
- **QLoRA** (`arXiv:2305.14314`): quantise the base model to 4-bit NF4, then apply LoRA adapters in full precision. Fits large models on a single consumer GPU for fine-tuning.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| PagedAttention — Kwon et al., 2023 | `arXiv:2309.06180` |
| QLoRA — Dettmers et al., 2023 | `arXiv:2305.14314` |
| LoRA — Hu et al., 2021 | `arXiv:2106.09685` |
| vLLM docs | docs.vllm.ai |

#### Interview questions
1. Why did batched inference with vLLM give you 20× throughput? Walk through the mechanism.
2. What's the difference between prefill and decode, and why does it matter for serving latency?
3. Explain LoRA's math. Why does low-rank work well for fine-tuning?
4. When would you choose QLoRA over full fine-tuning over prompt engineering?

---

### Module 7 🔵 — LLM internals

**Field knowledge every senior ML engineer needs.**

#### Concepts to know cold

- **Transformer architecture:** multi-head self-attention, feed-forward layers, layer norm, residual connections. Be able to describe the full forward pass.
- **Multi-head attention:** split queries, keys, values into h heads, compute attention in parallel, concatenate. Each head can attend to different aspects of the sequence.
- **Positional encodings:** sinusoidal (original), RoPE (Rotary Position Embedding — used by Llama, Mistral), ALiBi (attention with linear biases — better extrapolation to long sequences).
- **KV cache:** during inference, cache the key/value projections for all past tokens so you don't recompute them. Grows linearly with sequence length. This is what makes long context expensive.
- **Grouped Query Attention (GQA):** multiple query heads share a single key/value head. Reduces KV cache size by h/g× where g = number of KV groups. Used in Llama 2+, Mistral.
- **Mixture of Experts (MoE):** replace the FFN layer with N expert FFNs. Route each token to the top-k experts via a gating network. Only k experts activate per token — more parameters, same compute. Used in Mixtral, GPT-4 (speculated).
- **RLHF:** pre-train → supervised fine-tuning → reward model training → PPO optimisation. Reward model scores completions against human preferences. PPO updates the policy to maximise reward while staying close to the SFT model (KL penalty).
- **DPO** (`arXiv:2305.18290`): Direct Preference Optimisation. Reformulates RLHF to skip the reward model entirely — optimise the policy directly from preference pairs. Simpler, more stable, often as good as PPO.
- **Chinchilla scaling laws** (`arXiv:2203.15556`): for a given compute budget, optimal model size and training tokens are roughly equal. Prior models (GPT-3) were significantly over-parameterised relative to training data. Implication: smaller models trained longer often outperform larger undertrained ones.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| Attention is all you need — Vaswani et al. | `arXiv:1706.03762` |
| Chinchilla scaling laws | `arXiv:2203.15556` |
| DPO — Rafailov et al., 2023 | `arXiv:2305.18290` |
| Let's build GPT — Karpathy | youtube.com (search "Karpathy GPT from scratch") |
| Sebastian Raschka — LLM internals | sebastianraschka.com |

#### Interview questions
1. Explain how KV caching works and what determines its memory footprint.
2. What's the difference between RLHF and DPO? When would you use each?
3. Walk me through the Chinchilla finding and its practical implication for model training.
4. What is GQA and why does it matter for inference efficiency?

---

### Build 2 🟢 — Production RAG pipeline with evals (days 11–17)

**What:** Rebuild your Qdrant RAG pipeline with hybrid search, BGE re-ranking, and a full RAGAS eval suite.

**Why:** Your resume has the RAG bullet but no eval story. This closes that gap and produces something you can demo live.

**Stack:** Python, Qdrant, FastAPI, LangChain, HuggingFace (BGE reranker), RAGAS

**Milestones:**
1. Stand up Qdrant with both dense (text-embedding-3-small) and sparse (BM25) vectors
2. Implement Reciprocal Rank Fusion over dense + sparse results
3. Add BGE cross-encoder re-ranking as a second stage
4. Wire a FastAPI endpoint: query in → reranked chunks + answer out
5. Build a RAGAS eval suite: faithfulness, answer relevance, context precision, context recall
6. Blog: *"Dense vs sparse vs hybrid retrieval — a measured comparison with RAGAS"*

**Deliverables:**
- GitHub repo — end-to-end pipeline, runnable with Docker Compose
- FastAPI service with Swagger docs
- RAGAS eval notebook with ablation: dense-only vs hybrid vs hybrid+rerank
- Blog post: the retrieval ablation with real numbers

---

## Phase 3 — Evals, design principles & production (days 19–27)

> The two things that separate senior from mid-level in interviews. Evals is your flagged gap — don't just read about it, build a framework. AI systems design is what interviewers probe when they want to see if you can think beyond model quality.

---

### Module 8 🔴 — Evaluation & benchmarking

**Your gap. Prepare a clear POV — interviewers at frontier companies will probe this.**

#### Concepts to know cold

- **LLM-as-judge:** use a capable LLM (GPT-4o, Claude Sonnet) to score agent outputs against a rubric. Scales better than human eval, but inherits model biases. Calibrate against human labels before trusting it.
- **Failure modes of LLM-as-judge:** sycophancy (prefers its own outputs), verbosity bias (prefers longer answers), position bias (prefers first option in pairwise), self-enhancement bias. Mitigate with: swap positions, use multiple judges, use smaller targeted rubrics.
- **Component vs E2E evals:** test retrieval (precision@k, recall@k, MRR), generation (faithfulness, relevance, groundedness), and end-to-end task success separately. E2E evals are expensive — use them sparingly, for regression testing.
- **RAGAS metrics:** faithfulness, answer relevance, context precision, context recall. Know what each measures and what can go wrong.
- **Agent evals — the hard problem:** agents don't have a single right answer. Partial credit matters. Trajectory quality matters (did it take unnecessary steps?). Tool call precision matters (did it call the right tools?).
- **Rubric design:** define success criteria per task class. Break down into sub-criteria. Weight them. Use binary sub-criteria where possible (LLM-as-judge is less reliable on ordinal scales).
- **Memory-specific metrics:** preference capture rate, preference adherence rate, contradiction resolution accuracy. These are your personal contribution — know them cold.
- **Observability tools:** LangSmith (tracing, evaluation), Braintrust (production eval platform), DeepEval (open-source).
- **Regression testing:** every prompt change should run against a fixed eval suite before shipping. Store evals in version control alongside prompts.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| RAGAS | `arXiv:2309.15217` |
| SWE-bench | `arXiv:2310.06770` |
| DeepEval | github.com/confident-ai/deepeval |
| Braintrust | braintrust.dev |
| LangSmith | docs.smith.langchain.com |

#### Interview questions
1. How did you know Tastes was actually capturing preferences correctly? What would a proper eval look like?
2. What's the difference between faithfulness and relevance in RAG evals?
3. How do you eval an agent on a long-horizon task where partial success matters?
4. What are the failure modes of LLM-as-judge?
5. How would you set up a regression suite for a prompt change in production?

---

### Module 9 🔵 — AI systems design

**Scaling, reliability, and production patterns.**

#### Concepts to know cold

- **Rate limiting and backpressure:** LLM APIs have TPM/RPM limits. Design for graceful degradation — queue, backoff, fallback to cached responses.
- **Idempotency in agentic pipelines:** agent actions should be safe to retry. Use idempotency keys for tool calls. Store intermediate state so you can resume rather than restart.
- **Async agent execution:** long-horizon agents shouldn't block a request thread. Use a queue (Celery, Kafka, SQS) + worker pattern. Return a job ID immediately, poll or webhook for results.
- **Cost optimisation:** prompt caching (Anthropic/OpenAI) for static context. Model routing — use a smaller/cheaper model for simple subtasks, escalate to a frontier model only when needed. Batching for offline workloads.
- **Structured output reliability:** JSON mode reduces but doesn't eliminate parsing failures. Use Instructor or Pydantic models + retry-on-parse-failure. Have a fallback schema for degraded cases.
- **Guardrails:** input validation (topic classifiers, injection detection), output validation (schema check, toxicity, factual grounding), rate limiting per user.
- **Human-in-the-loop patterns:** approval gates for irreversible actions, async escalation for low-confidence decisions, audit logs for all agent actions.
- **Multi-tenancy:** per-user memory isolation, per-tenant rate limits, separate vector collections or namespace prefixes per tenant.
- **Observability:** distributed traces (one span per agent step), token usage logging per step, latency breakdowns (retrieval vs generation vs tool call), cost attribution per request.
- **12-factor agents:** stateless execution, external state stores, structured outputs, minimal footprint, pause/resume support.

#### Resources

| Resource | Link |
|---|---|
| 12-factor agents | github.com/humanlayer/12-factor-agents |
| AI Engineering — Chip Huyen | oreilly.com/library/view/ai-engineering |
| Prompt caching docs | docs.anthropic.com/en/docs/build-with-claude/prompt-caching |
| Instructor library | python.useinstructor.com |

#### Interview questions
1. How do you design an agentic system that's safe to retry on failure?
2. Walk me through how you'd reduce LLM API costs by 50% in a production agent.
3. How do you implement human-in-the-loop without making the UX terrible?
4. What does good observability look like for a multi-agent system?
5. How would you design multi-tenancy for an agent that has per-user memory?

---

### Build 3 🔴 — Agent eval framework (days 19–26)

**What:** A lightweight harness for evaluating multi-step agent trajectories. Your answer to the eval gap.

**Why:** Building a framework means you can speak to evals from first principles in interviews, not just theory. Pick a domain you know: IT management or workplace copilot tasks.

**Stack:** Python, LangGraph, Anthropic API, LangSmith, FastAPI, SQLite (trace storage)

**Milestones:**
1. Define a task schema: task description, expected tool calls, success criteria, partial credit rubric
2. Build a trajectory recorder that captures every agent step (thought, tool call, observation)
3. Implement LLM-as-judge scoring against the rubric — with calibration checks against human labels
4. Add component-level metrics: tool call precision, step efficiency, loop detection rate
5. Run against a suite of 20 IT management tasks — publish results
6. Blog: *"How to evaluate agents that don't have a single right answer"*

**Deliverables:**
- GitHub repo — pluggable eval harness, works with any LangGraph agent
- 20-task benchmark suite with ground truth annotations
- Results dashboard — per-task scores, failure mode breakdown
- Blog: designing rubrics for partial-credit agent evaluation

---

## Phase 4 — Frontier topics & staying sharp (days 28–35)

> These modules keep you sharp in the field regardless of interviews. The two builds here are the highest-upside portfolio pieces — genuinely novel, publishable work.

---

### Module 10 🔵 — Test-time compute & reasoning models

#### Concepts to know cold

- **Test-time compute scaling:** instead of scaling model size or training data, allocate more compute at inference time — more thinking steps, more candidates, more verification. Complementary to train-time scaling.
- **Chain-of-thought as internal reasoning:** CoT traces are thinking tokens that don't appear in the final output. Extended thinking (Claude, o1) makes this a first-class API feature with configurable token budgets.
- **Process reward models (PRM) vs outcome reward models (ORM):** ORM scores the final answer. PRM scores each reasoning step. PRMs enable more targeted feedback but are expensive to train (need step-level annotations).
- **MCTS for LLM reasoning:** Monte Carlo Tree Search explores multiple reasoning paths, uses a value function to prune, backtracks on dead ends. Produces better answers on hard problems at significant compute cost.
- **Budget forcing / thinking token limits:** cap the thinking budget per task class. Don't give a reasoning model unlimited tokens for a simple lookup — the extra thinking tokens won't help and will waste money.
- **When to use a reasoning model:** multi-step mathematical reasoning, ambiguous planning tasks, tasks where the answer isn't obvious from pattern matching. Don't use them for simple retrieval, classification, or extraction.
- **Best-of-N sampling:** generate N answers, pick the best by some scoring function (reward model, majority vote, LLM-as-judge). Simple but effective — often competitive with more complex MCTS approaches.
- **DeepSeek-R1** (`arXiv:2501.12948`): open reasoning model trained with GRPO (group relative policy optimisation), a PPO variant that avoids the need for a separate critic model. Competitive with o1 on math/coding benchmarks.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| DeepSeek-R1 | `arXiv:2501.12948` |
| Scaling LLM test-time compute — Google | `arXiv:2408.03314` |
| Anthropic extended thinking docs | docs.anthropic.com |

#### Interview questions
1. What's the difference between scaling train-time and test-time compute?
2. When would you use a reasoning model (o3, R1) in an agent vs a faster model?
3. What is a process reward model and why is it hard to train?
4. How does extended thinking change the way you design prompts?

---

### Module 11 🟢🔵 — Multimodal & long-context agents

**Your work (FNFI):** replaced text-only GRU models with LayoutLMv3 for document extraction. Deployed multimodal pipeline on AWS.

#### Concepts to know cold

- **LayoutLM family:** incorporates 2D position embeddings alongside text tokens — the model sees where on the page a text token appears. LayoutLMv3 also processes image patches. Critical for visually complex documents (forms, invoices, tables).
- **Vision encoders:** CLIP (contrastive image-text pre-training), SigLIP (sigmoid loss variant, better for zero-shot). Used in multimodal LLMs to encode images into token-compatible representations.
- **Long-context vs RAG tradeoff:** at 1M+ context windows, you can sometimes stuff everything in rather than retrieving. But: cost scales quadratically (attention), "lost in the middle" degradation for facts far from ends of context, no latency benefit over RAG for known subsets.
- **Needle-in-a-haystack benchmarks:** tests model's ability to retrieve a specific fact from a very long context. Modern frontier models perform well but still degrade in the middle of very long contexts.
- **OCR vs end-to-end multimodal:** traditional OCR → text extraction → NLP. End-to-end multimodal: model sees image directly, extracts structure and content jointly. End-to-end wins on visually complex layouts; OCR pipelines are more interpretable and debuggable.

#### Papers & resources

| Resource | ID / Link |
|---|---|
| LayoutLMv3 | `arXiv:2204.08387` |
| Anthropic vision docs | docs.anthropic.com/en/docs/build-with-claude/vision |

#### Interview questions
1. When did LayoutLM outperform text-only models at FNFI? What was the core advantage?
2. When would you choose a 1M context window over RAG? What are the cost/quality tradeoffs?
3. How do you evaluate a multimodal extraction pipeline vs a text-only one?

---

### Build 4 🟢 — Memory system benchmark (days 26–32)

**What:** Compare mem0, MemGPT (Letta), and Tasty on LOCOMO — a publishable head-to-head.

**Why:** You've read all the memory papers. Actually running a benchmark produces something genuinely novel. It also forces you to deeply understand each system's internals — which you'll need to defend in interviews. Clean public comparisons like this get traction in the agent/memory space.

**Stack:** Python, mem0, Letta (MemGPT), Anthropic API, LOCOMO dataset, pandas, matplotlib

**Milestones:**
1. Set up the LOCOMO benchmark harness — load the dataset, define the eval loop
2. Implement a Tasty adapter that conforms to the LOCOMO query interface
3. Run mem0 and MemGPT (Letta) on the same LOCOMO splits
4. Add your own metrics: preference capture rate, contradiction resolution rate, latency per memory op
5. Analyse failure modes — what does each system get wrong and why?
6. Blog: *"Benchmarking three LLM memory systems on LOCOMO — what the papers don't tell you"*

**Deliverables:**
- GitHub repo — reproducible benchmark with one-command setup
- Results notebook: side-by-side on LOCOMO + your custom metrics
- Blog: honest analysis of where each system wins and loses

---

### Build 5 🟢 — Safe code-gen agent with sandboxed execution (days 28–33)

**What:** Generalise your Postgres query pattern into a proper sandboxed code-gen framework.

**Why:** Your code-gen bullet is your most distinctive resume item but interviewers will immediately ask "how did you sandbox it?" This build produces a defensible, demo-able answer.

**Stack:** Python, LangGraph, Docker SDK, FastAPI, Anthropic API, HuggingFace (code eval)

**Milestones:**
1. Build a Docker-based execution sandbox: no network, no filesystem, resource-limited, timeout-enforced
2. Implement an SDK-indexing layer: parse a Python SDK into a context-friendly doc the LLM can query
3. Build the code-gen agent loop: generate → validate → sandbox-execute → observe → retry
4. Add safety checks: static analysis before execution, output schema validation after
5. Benchmark vs traditional tool-calling on a query suite: latency, accuracy, failure rate
6. Blog: *"Code generation as tool calling — the architecture, the risks, and how to sandbox it"*

**Deliverables:**
- GitHub repo — full framework, Docker Compose, example SDK integration
- FastAPI service: submit a natural language query, get back code + result
- Benchmark notebook: code-gen vs tool-calling on 30 representative queries
- Blog: sandboxing strategy, static analysis gates, the retry loop design

---

### Build 6 🔵 — Reasoning-augmented agent with test-time compute routing (days 32–35+)

**What:** An agent that dynamically routes between fast (Sonnet) and slow (extended thinking) models by task complexity.

**Why:** Test-time compute is the frontier topic every interviewer at a top AI company will probe. Building a model router that dynamically allocates thinking budget — and measuring when it actually helps — puts you well ahead of candidates who've only read about it.

**Stack:** Python, LangGraph, Anthropic API (Sonnet + extended thinking), FastAPI, LangSmith

**Milestones:**
1. Build a complexity classifier: route tasks to fast (Sonnet) or slow (extended thinking) model
2. Implement a budget-forcing mechanism: cap thinking tokens per task class
3. Design an eval suite across three task classes — retrieval, multi-step reasoning, ambiguous planning
4. Measure: accuracy delta, latency delta, cost delta for fast vs slow routing
5. Build a routing policy based on the results — when does extended thinking actually help?
6. Blog: *"When does extended thinking actually help? — a measured answer with a real agent"*

**Deliverables:**
- GitHub repo — router + agent + eval harness
- Results: accuracy/latency/cost across task classes, fast vs slow model
- Blog: the routing policy, the surprising results, and the design implications

---

## Reference: all papers

| Paper | arXiv ID |
|---|---|
| ReAct — Yao et al., 2022 | 2210.03629 |
| Plan-and-Solve prompting | 2305.04091 |
| CoALA — cognitive architectures for agents | 2309.02427 |
| Survey on LLM-based autonomous agents | 2308.11432 |
| PAL — Program-Aided Language Models | 2211.10435 |
| ToolBench — tool learning survey | 2304.08354 |
| RAG — Lewis et al., 2020 | 2005.11401 |
| RAGAS | 2309.15217 |
| Corrective RAG (CRAG) | 2401.15884 |
| MemGPT — Packer et al., 2023 | 2310.08560 |
| PAMU — preference-aware memory | 2510.09720 |
| Graphiti/Zep — bi-temporal KG | 2501.13956 |
| mem0 — multi-signal memory | 2504.19413 |
| Anatomy of agentic memory | 2602.19320 |
| PERMA benchmark | 2603.23231 |
| A-MEM — Zettelkasten agentic memory | 2502.12110 |
| Memori — persistent memory layer | 2603.19935 |
| Memoria — KG user modelling | 2512.12686 |
| Memory for Autonomous LLM Agents (survey) | 2603.07670 |
| From Experience to Strategy: Graph Memory | 2511.07800 |
| MemoryAgentBench (ICLR 2026) | 2507.05257 |
| PagedAttention — vLLM | 2309.06180 |
| LoRA — Hu et al., 2021 | 2106.09685 |
| QLoRA — Dettmers et al., 2023 | 2305.14314 |
| Attention is all you need | 1706.03762 |
| Chinchilla scaling laws | 2203.15556 |
| DPO — Rafailov et al., 2023 | 2305.18290 |
| SWE-bench | 2310.06770 |
| DeepSeek-R1 | 2501.12948 |
| Scaling LLM test-time compute | 2408.03314 |
| Agent-FLAN | 2403.12881 |
| LayoutLMv3 | 2204.08387 |

---

## Reference: channels & people to follow

- **Andrej Karpathy** — `@karpathy` on X — context engineering, agent design
- **Harrison Chase** — LangChain CEO — agent patterns, context engineering
- **Charles Packer** — Letta/MemGPT creator — memory systems
- **The Sequence** — newsletter — weekly roundup of AI engineering papers and products
- **Latent Space podcast** — deep technical interviews with AI engineers
- **Interconnects** — Nathan Lambert's newsletter — RL, fine-tuning, agent training
- **Ahead of AI** — Sebastian Raschka — ML papers explained clearly

---

## Reference: recommended books

- *Designing Machine Learning Systems* — Chip Huyen — best practical ML systems book
- *AI Engineering* — Chip Huyen (2025) — LLM application engineering specifically
- *Building LLM Powered Applications* — Valentina Alto — RAG, agents, memory end-to-end

---

*Last updated: May 2026. Built for Viraj Singh — Skyfall AI ML Engineer.*

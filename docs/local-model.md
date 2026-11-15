# Running the local model (Qwen3 8B)

JARVIS uses a self-hosted [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B)
by default. The model is free to download and run (Apache 2.0); you pay
nothing per message. What you need is a machine that runs it and a small
local server that exposes an OpenAI-compatible endpoint, which is what
`JARVIS_LOCAL_LLM_BASE_URL` points at.

Qwen3 8B is a dense 8.2B-parameter model with a 32K-token context (131K
with YaRN) and native tool calling, which is what the agent loop uses.

## Hardware guideline

These are recommended targets based on published build sizes, not verified
minimums. We have not tested specific hardware, so treat them as starting
points, not a guaranteed fit.

- An Apple Silicon Mac with 16 GB of unified memory is the commonly
  recommended target for the 4-bit builds (about a 5 GB download).
- On a Linux/Windows PC or server, quantized builds are generally run on
  a GPU with about 12 GB VRAM; full precision wants about 20 GB.
- Smaller or slower machines may still work with more aggressive
  quantization; answers just take longer. Verify on your own hardware.

## Option A: LM Studio (easiest, Mac)

1. Install LM Studio from https://lmstudio.ai
2. In the app, search for and download `qwen3-8b` (the MLX 4-bit build
   is usually fastest on Apple Silicon).
3. Start the local server (Developer tab, default port 1234).
4. In `.env`:

   ```
   JARVIS_LLM_PROVIDER=local
   JARVIS_LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
   JARVIS_LOCAL_LLM_MODEL=qwen3-8b
   ```

## Option B: Ollama (terminal, Mac/Linux/Windows)

```bash
ollama pull qwen3:8b
ollama serve   # listens on port 11434
```

```
JARVIS_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
JARVIS_LOCAL_LLM_MODEL=qwen3:8b
```

## Option C: mlx-lm (Apple Silicon, scriptable)

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080
```

```
JARVIS_LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
JARVIS_LOCAL_LLM_MODEL=mlx-community/Qwen3-8B-4bit
```

## Option D: vLLM (server with a GPU, for real users)

```bash
vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

```
JARVIS_LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
JARVIS_LOCAL_LLM_MODEL=Qwen/Qwen3-8B
```

## Notes

- Web search is separate. The model cannot browse the web by itself, so
  fresh research also needs `EXA_API_KEY` (see `.env.example`). Without
  it, research reports itself unconfigured instead of guessing.
- No fine-tuning is required: Qwen3 8B is already instruction-tuned and
  trained for tool calling.
- Thinking vs fast mode: Qwen3 can reason step by step or answer
  directly. LM Studio and vLLM expose this per request; the defaults are
  fine to start.
- Hosted OpenAI remains available as an explicit opt-in:
  `JARVIS_LLM_PROVIDER=openai` plus `OPENAI_API_KEY`.

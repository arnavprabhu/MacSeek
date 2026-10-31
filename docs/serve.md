# Chat server + web UI

    tools/serve.sh                 # builds if needed, loads the model (~10 s), serves http://127.0.0.1:8080
    PORT=9000 HOST=0.0.0.0 tools/serve.sh --max-tokens 2048

- Web chat: http://127.0.0.1:8080/ (web/index.html, read per request: edit + reload).
- OpenAI-compatible: POST /v1/chat/completions (stream or not, tools / tool_calls, reasoning_content),
  POST /v1/completions, GET /v1/models, /health, /props (supported params), /v1/stats (live engine state).
- Sampling params: temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty
  (alias repeat_penalty), repeat_last_n, seed, max_tokens / max_completion_tokens, stop, logit_bias, ignore_eos.
  Template: reasoning_effort (low|high|max|1-100), chat_template_kwargs.thinking_mode (thinking|chat).
- Responses carry llama.cpp-style `timings` (prompt_n, cache_n, prompt_per_second, predicted_per_second).
- Prefix reuse: a follow-up turn only prefills the new tokens (snapshot after the last <｜Assistant｜>).
- One model process at a time (tools/modellock.sh); stop it with Ctrl+C before running benchmarks.

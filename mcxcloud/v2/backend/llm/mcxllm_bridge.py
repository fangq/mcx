#!/usr/bin/env python3
"""Bridge between the Fastify API and MCXLLM (natural-language prompt -> MCX input JSON).

Runs under the interpreter that has MCXLLM's dependencies (langchain, openai, pydantic),
which is NOT the API's runtime — see LLM_PYTHON / LLM_MCX_DIR in backend/.env. The API
spawns one process per request (src/llm.js) and speaks a JSON envelope over stdin/stdout:

    in :  {"prompt": "...", "backend": "openai"|"ollama", "model": "gpt-4o", "apiKey": "sk-..."}
    out:  {"ok": true,  "doc": {...}, "elapsed": 6.4, "backend": "...", "model": "..."}
          {"ok": false, "error": "...", "raw": "..."}

MCXLLM is imported from LLM_MCX_DIR rather than vendored, so it stays in step with that
project. Selecting a backend rebinds module globals that `LLMMCX` resolves at call time
(`client`, `get_completion`), so no file there is edited:

  openai — a fresh OpenAI client on the user's key; the model name is pinned.
  ollama — `get_completion` is replaced by a direct POST to LLM_OLLAMA_URL, the full
           OpenAI-style chat endpoint (e.g. https://neurojson.io/api/v1/ollama-public/chat,
           or http://host:11434/v1/chat/completions; a bare ".../v1" gets the standard path
           appended). The OpenAI SDK is not used here because it always appends
           /chat/completions to its base URL, and proxies serve the same API elsewhere.

The openai backend runs on the key in the envelope (the user's own) and nothing else: the
API strips every OPENAI_* variable from this process's environment, so a server-side key
can never be picked up by accident.

stdout is redirected to stderr for the duration of the import and the call: langchain and
openai are free to print there, and any stray byte would corrupt the envelope.
"""

import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

MAX_RAW = (
    4000  # cap the echoed raw model output so a runaway reply can't bloat the reply
)
HTTP_TIMEOUT = (
    600  # socket timeout for the ollama call; the API's LLM_TIMEOUT_MS kills us first
)


def chat_url(url):
    """LLM_OLLAMA_URL as a full chat endpoint; a bare OpenAI-style '.../v1' base is completed."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("LLM_OLLAMA_URL is unset")
    return url + "/chat/completions" if url.endswith("/v1") else url


def chat_completion(url, model, prompt):
    """One non-streaming OpenAI-style chat call; returns the assistant's text.

    Same sampling as MCXLLM's own get_completion (temperature 0, top_p 0), so the two
    backends differ only in the model.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "mcxcloud-llm-bridge",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as res:
            reply = json.load(res)
    except urllib.error.HTTPError as err:
        detail = err.read(500).decode("utf-8", "replace")
        raise RuntimeError(
            "%s answered HTTP %d: %s" % (url, err.code, detail)
        ) from None
    except urllib.error.URLError as err:
        raise RuntimeError("cannot reach %s: %s" % (url, err.reason)) from None

    if isinstance(reply, dict) and reply.get("error"):
        e = reply["error"]
        raise RuntimeError(
            "model error: %s" % (e.get("message") if isinstance(e, dict) else e)
        )
    try:
        choice = reply["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError(
            "unexpected reply shape from %s: %s" % (url, str(reply)[:300])
        ) from None
    if not content.strip():
        # a reasoning model that ran out of tokens mid-thought returns only `reasoning`
        raise RuntimeError(
            "the model returned an empty answer (finish_reason=%s)"
            % choice.get("finish_reason")
        )
    return content


def emit(ok, **fields):
    """Write the one-and-only stdout line. Must run with stdout un-redirected."""
    json.dump({"ok": ok, **fields}, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as err:
        emit(False, error="unreadable request envelope: %s" % err)
        return 1

    prompt = (req.get("prompt") or "").strip()
    backend = req.get("backend") or "openai"
    model = req.get("model") or ""
    api_key = (req.get("apiKey") or "").strip()
    if not prompt:
        emit(False, error="empty prompt")
        return 0
    if backend == "openai" and not api_key:
        emit(False, error="the openai backend needs the user's API key")
        return 1

    mcx_dir = os.environ.get("LLM_MCX_DIR", "")
    if not mcx_dir or not os.path.isdir(mcx_dir):
        emit(False, error="LLM_MCX_DIR is unset or not a directory: %r" % mcx_dir)
        return 1
    sys.path.insert(0, mcx_dir)

    started = time.time()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            # LLMMCX_gpt4o builds `client = OpenAI()` at import time, which raises without
            # OPENAI_API_KEY. That client is replaced below before any call is made, so a
            # placeholder is enough to get through the import — it never reaches OpenAI.
            os.environ["OPENAI_API_KEY"] = "placeholder-replaced-before-use"
            # LLMMCX_gpt4o itself imports LLMJson_post_gpt4o from the same directory.
            import LLMMCX_gpt4o as mcxllm
            from openai import OpenAI

            if backend == "ollama":
                url = chat_url(os.environ.get("LLM_OLLAMA_URL", ""))
                # LLMMCX calls get_completion(prompt); route it straight to the endpoint.
                mcxllm.get_completion = lambda p, m=None: chat_completion(url, model, p)
            elif backend == "openai":
                mcxllm.client = OpenAI(api_key=api_key)
                if model:
                    # LLMMCX calls get_completion(prompt) with no model argument, so pin it
                    # here rather than edit the signature upstream.
                    _orig = mcxllm.get_completion
                    mcxllm.get_completion = lambda p, m=model: _orig(p, m)
            else:
                raise RuntimeError("unknown backend %r" % backend)

            result = mcxllm.LLMMCX(prompt)
    except Exception as err:  # import failure, missing API key, host unreachable, ...
        emit(
            False,
            error="%s: %s" % (type(err).__name__, err),
            elapsed=round(time.time() - started, 2),
            backend=backend,
            model=model,
        )
        return 1

    elapsed = round(time.time() - started, 2)
    meta = {"elapsed": elapsed, "backend": backend, "model": model}

    # LLMMCX returns a JSON string on success, or the literal string "Error: ..." when its
    # own post-processing raised (see the except branch at the bottom of LLMMCX_gpt4o.py).
    if not isinstance(result, str):
        emit(
            False,
            error="MCXLLM returned %s, expected a JSON string" % type(result).__name__,
            **meta
        )
        return 1
    if result.startswith("Error:"):
        emit(False, error=result, **meta)
        return 1

    try:
        doc = json.loads(result)
    except json.JSONDecodeError as err:
        emit(
            False,
            error="MCXLLM did not return valid JSON: %s" % err,
            raw=result[:MAX_RAW],
            **meta
        )
        return 1

    emit(True, doc=doc, **meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

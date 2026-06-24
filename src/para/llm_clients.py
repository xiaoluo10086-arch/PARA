"""LLM API helpers for PARA experiments."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Mapping, Optional


def is_local_endpoint(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def is_gemini_endpoint(base_url: str, model: str) -> bool:
    lowered = (base_url + " " + model).lower()
    return "generativelanguage.googleapis.com" in lowered or base_url.startswith("gemini://") or model.startswith("gemini-")


def is_deepseek_endpoint(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    return "deepseek" in (parsed.hostname or "").lower()


def api_headers(base_url: str, model: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    parsed = urllib.parse.urlparse(base_url if not base_url.startswith("gemini://") else "https://generativelanguage.googleapis.com")
    host = parsed.hostname or ""
    if is_gemini_endpoint(base_url, model):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("PARA_GEMINI_API_KEY") or os.getenv("ASHRL_GEMINI_API_KEY")
        if key:
            headers["x-goog-api-key"] = key
        return headers
    if "deepseek" in host:
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("PARA_DEEPSEEK_API_KEY") or os.getenv("ASHRL_DEEPSEEK_API_KEY")
    elif "openai" in host:
        key = os.getenv("OPENAI_API_KEY") or os.getenv("PARA_OPENAI_API_KEY") or os.getenv("ASHRL_OPENAI_API_KEY")
    else:
        key = os.getenv("PARA_LLM_API_KEY") or os.getenv("NSHRL_LLM_API_KEY") or os.getenv("ASHRL_LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def chat_text(
    *,
    base_url: str,
    model: str,
    messages: List[Mapping[str, str]],
    request_timeout: int,
    temperature: float,
    max_tokens: int,
    json_response: bool = True,
    grammar: Optional[str] = None,
    response_format: Optional[Mapping[str, object]] = None,
    extra_payload: Optional[Mapping[str, object]] = None,
) -> str:
    """Return chat text from either OpenAI-compatible or Gemini API."""

    if is_gemini_endpoint(base_url, model):
        return gemini_chat_text(
            base_url=base_url,
            model=model,
            messages=messages,
            request_timeout=request_timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            json_response=json_response,
        )
    if (
        is_deepseek_endpoint(base_url)
        and response_format is not None
        and response_format.get("type") == "json_schema"
    ):
        return deepseek_strict_tool_chat_text(
            base_url=base_url,
            model=model,
            messages=messages,
            request_timeout=request_timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_payload=extra_payload,
        )
    return openai_compatible_chat_text(
        base_url=base_url,
        model=model,
        messages=messages,
        request_timeout=request_timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        json_response=json_response,
        grammar=grammar if is_local_endpoint(base_url) else None,
        response_format=response_format,
        extra_payload=extra_payload,
    )


def openai_compatible_chat_text(
    *,
    base_url: str,
    model: str,
    messages: List[Mapping[str, str]],
    request_timeout: int,
    temperature: float,
    max_tokens: int,
    json_response: bool,
    grammar: Optional[str],
    response_format: Optional[Mapping[str, object]] = None,
    extra_payload: Optional[Mapping[str, object]] = None,
) -> str:
    payload: Dict[str, object] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    if response_format is not None:
        payload["response_format"] = dict(response_format)
    elif json_response:
        payload["response_format"] = {"type": "json_object"}
    if grammar:
        payload["grammar"] = grammar
        payload["grammar_string"] = grammar
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = api_headers(base_url, model)
    raw = post_json(url, payload, headers, request_timeout)
    message = raw["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def deepseek_strict_tool_chat_text(
    *,
    base_url: str,
    model: str,
    messages: List[Mapping[str, str]],
    request_timeout: int,
    temperature: float,
    max_tokens: int,
    response_format: Mapping[str, object],
    extra_payload: Optional[Mapping[str, object]] = None,
) -> str:
    """Return JSON arguments from DeepSeek beta strict function calling."""

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        raise ValueError("DeepSeek strict tool mode requires response_format.json_schema")
    name = str(json_schema.get("name") or "emit_action")
    schema = json_schema.get("schema")
    if not isinstance(schema, Mapping):
        raise ValueError("DeepSeek strict tool mode requires response_format.json_schema.schema")

    payload: Dict[str, object] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Emit the PARA agent action object.",
                    "strict": True,
                    "parameters": dict(schema),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }
    if extra_payload:
        for key, value in extra_payload.items():
            if key not in {"response_format", "tools", "tool_choice"}:
                payload[key] = value

    parsed = urllib.parse.urlparse(base_url)
    api_root = f"{parsed.scheme or 'https'}://{parsed.netloc or 'api.deepseek.com'}"
    url = api_root.rstrip("/") + "/beta/v1/chat/completions"
    raw = post_json(url, payload, api_headers(base_url, model), request_timeout)
    message = raw["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return str(message.get("content") or message.get("reasoning_content") or "")
    arguments = tool_calls[0]["function"]["arguments"]
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def gemini_chat_text(
    *,
    base_url: str,
    model: str,
    messages: List[Mapping[str, str]],
    request_timeout: int,
    temperature: float,
    max_tokens: int,
    json_response: bool,
) -> str:
    api_root = "https://generativelanguage.googleapis.com/v1beta"
    if base_url and not base_url.startswith("gemini://"):
        api_root = base_url.rstrip("/")
    model_name = model if model.startswith("models/") else "models/" + model
    system_parts = []
    user_parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append({"text": content})
        else:
            user_parts.append({"text": content})
    gemini_max_tokens = max_tokens
    min_output_tokens = int(os.getenv("ASHRL_GEMINI_MIN_OUTPUT_TOKENS", "0") or "0")
    if min_output_tokens > 0:
        gemini_max_tokens = max(gemini_max_tokens, min_output_tokens)
    generation_config: Dict[str, object] = {"temperature": temperature, "maxOutputTokens": gemini_max_tokens}
    thinking_budget = os.getenv("ASHRL_GEMINI_THINKING_BUDGET")
    if thinking_budget not in {None, ""}:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

    payload: Dict[str, object] = {
        "contents": [{"role": "user", "parts": user_parts or [{"text": ""}]}],
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}
    if json_response:
        payload["generationConfig"]["responseMimeType"] = "application/json"  # type: ignore[index]
    endpoint = f"{api_root}/{urllib.parse.quote(model_name, safe='/')}:generateContent"
    attempts = max(1, 1 + int(os.getenv("ASHRL_LLM_EMPTY_RESPONSE_RETRIES", "0") or "0"))
    last_text = ""
    for attempt in range(attempts):
        raw = post_json(endpoint, payload, api_headers(base_url, model), request_timeout)
        candidates = raw.get("candidates") or []
        if candidates:
            parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
            text = "\n".join(str(part.get("text", "")) for part in parts)
            if text.strip():
                return text
            last_text = text
        if attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
    return last_text


def post_json(url: str, payload: Mapping[str, object], headers: Mapping[str, str], request_timeout: int) -> Dict[str, object]:
    if use_curl_transport():
        return post_json_curl(url, payload, headers, request_timeout)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def use_curl_transport() -> bool:
    if os.getenv("ASHRL_LLM_TRANSPORT", "").lower() == "curl":
        return True
    proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy") or ""
    return proxy.lower().startswith("socks")


def post_json_curl(url: str, payload: Mapping[str, object], headers: Mapping[str, str], request_timeout: int) -> Dict[str, object]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as payload_file:
        json.dump(payload, payload_file)
        payload_path = payload_file.name
    config_lines = [
        f'url = "{url}"',
        'request = "POST"',
        f"max-time = {int(request_timeout)}",
        "silent",
        "show-error",
        "fail-with-body",
        f'data-binary = "@{payload_path}"',
    ]
    for key, value in headers.items():
        config_lines.append(f'header = "{key}: {value}"')
    proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or ""
    if proxy:
        parsed = urllib.parse.urlparse(proxy)
        if parsed.scheme in {"socks5", "socks5h"}:
            config_lines.append(f'socks5-hostname = "{parsed.hostname}:{parsed.port or 1080}"')
        else:
            config_lines.append(f'proxy = "{proxy}"')
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as config_file:
        config_file.write("\n".join(config_lines))
        config_path = config_file.name
    try:
        proc = None
        for attempt in range(3):
            proc = subprocess.run(
                ["curl", "--config", config_path],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=request_timeout + 10,
                check=False,
            )
            if proc.returncode == 0 or proc.returncode == 22:
                break
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        assert proc is not None
        if proc.returncode != 0:
            raise RuntimeError(f"curl transport failed rc={proc.returncode}: {proc.stderr[:500]} {proc.stdout[:500]}")
        return json.loads(proc.stdout)
    finally:
        for path in (payload_path, config_path):
            try:
                os.unlink(path)
            except OSError:
                pass

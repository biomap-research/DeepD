#!/usr/bin/env python3
"""
Async inference gateway CLI for DNA vLLM jobs.

Flow (new job):
  1) POST https://ideal.biomap.com/api/online/inference -> task_id
  2) Poll GET .../inference?task_id=... until success
  3) Download {task_id}.json; print generate text / logits ppl / embedding shape

Flow (history job):
  python cli.py --task-id <task_id> [--task-type embedding] --output-dir ./results

Examples:
  python cli.py --prompt AT --task-type embedding --species "Homo sapiens"
  python cli.py --prompt AT --task-type embedding
  python cli.py --prompt AT --task-type generate --max-tokens 64 --top-k 1 --species "Argentina anserina"
  python cli.py --prompt-file seq.fa --task-type logits --species "Argentina anserina"
  python cli.py --task-id 0123456789abcdef0123456789abcdef --output-dir ./results

Species are requested by their scientific name (--species); cli.py resolves
the name to the internal species id the gateway expects, so callers never
need to handle genome / assembly accessions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from species import (  # noqa: E402
    SPECIES_RAW_EMPTY,
    SpeciesError,
    species_raw_is_unset,
    to_species_raw,
)

DEFAULT_SUBMIT_FILE = "etc/submit_job.json"
DEFAULT_GATEWAY_URL = "https://ideal.biomap.com/api/online/inference"
DEFAULT_TOKEN_ENV = "INFERENCE_API_TOKEN"
DEFAULT_SUCCESS_STATUSES = (
    "success",
    "succeeded",
    "completed",
    "done",
    "finished",
)
DEFAULT_FAILURE_STATUSES = (
    "failed",
    "error",
    "cancelled",
    "canceled",
    "timeout",
    "killed",
)
DEFAULT_RUNNING_STATUSES = (
    "pending",
    "queued",
    "submitted",
    "running",
    "in_progress",
    "processing",
)

MAX_MODEL_LEN = 10000
# Gateway always receives parameters.species_raw (internal species id or the
# SPECIES_RAW_EMPTY sentinel). The client never sends species_token_id.
RESERVED_PROMPT_TOKENS = 3  # species / special tokens prepended on worker
MAX_PROMPT_LENGTH = MAX_MODEL_LEN - RESERVED_PROMPT_TOKENS

# Legacy top-level keys that must live inside parameters.generate_params_json.
# Value sent in parameters.generate_params_json when no generate flags are given.
DEFAULT_GENERATE_PARAMS_JSON_VALUE = "'{}'"

_GENERATE_FLAT_KEYS = (
    "max_tokens",
    "min_tokens",
    "top_p",
    "top_k",
    "min_p",
    "temperature",
    "ignore_eos",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str, path: Path) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError(f"empty submit file: {path}")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # curl example in etc/submit_job.json: JSON between first `{` after `-d` and last `}`
    if "-d" in text:
        anchor = text.find("-d")
        start = text.find("{", anchor)
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"cannot parse JSON body from {path}")


def _normalize_bearer_token(token: str) -> str:
    value = token.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def _read_api_key_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and all(ord(c) < 128 for c in line):
            return line
    return None


def _resolve_bearer_token(args: argparse.Namespace) -> str:
    raw = args.token or os.environ.get(DEFAULT_TOKEN_ENV) or os.environ.get(
        "DNA_INFERENCE_TOKEN"
    )
    if not raw or not str(raw).strip():
        apiexample = Path(__file__).resolve().parent
        for candidate in (apiexample.parent / "API-Key.txt", apiexample / "API-Key.txt"):
            raw = _read_api_key_file(candidate)
            if raw:
                break
    if not raw or not str(raw).strip():
        raise ValueError(
            f"API token is required (--token, env {DEFAULT_TOKEN_ENV}, "
            "or API-Key.txt at the repository root)"
        )
    return _normalize_bearer_token(str(raw))


def _build_request_headers(token: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _submit_job(
    gateway_url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    token: str,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        gateway_url,
        data=data,
        method="POST",
        headers=_build_request_headers(token, json_body=True),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise TypeError(f"submit response must be object, got {type(out)!r}")
    return out


def _query_job(
    query_url: str,
    timeout: float,
    *,
    token: str,
) -> dict[str, Any]:
    req = urllib.request.Request(
        query_url,
        method="GET",
        headers=_build_request_headers(token),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise TypeError(f"query response must be object, got {type(out)!r}")
    return out


def _normalize_status(status: Any) -> Optional[str]:
    if status is None:
        return None
    return str(status).strip().lower()


def _get_result_block(resp: dict[str, Any]) -> dict[str, Any]:
    """Gateway wraps job fields in top-level ``result`` (see etc/*_response.json)."""
    block = resp.get("result")
    if isinstance(block, dict):
        return block
    return resp


def _ensure_api_ok(resp: dict[str, Any], *, context: str) -> None:
    code = resp.get("code")
    if code is None:
        return
    if int(code) == 200:
        return
    msg = resp.get("message")
    raise RuntimeError(f"{context} API error: code={code}, message={msg!r}")


def _decode_query_error_body(body: str) -> tuple[Optional[dict[str, Any]], str]:
    text = body.strip()
    if not text:
        return None, text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, text
    except json.JSONDecodeError:
        pass
    return None, text


def _format_query_api_error(
    *,
    http_code: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
    raw_body: str = "",
) -> str:
    if payload is not None:
        api_code = payload.get("code")
        msg = payload.get("message")
        if api_code is not None or msg:
            return f"query failed: code={api_code}, message={msg!r}"
    if http_code is not None:
        snippet = raw_body.strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        return f"query failed: HTTP {http_code}: {snippet}"
    return f"query failed: {raw_body.strip()[:500]}"


def _should_retry_query_http_error(
    http_code: int,
    payload: Optional[dict[str, Any]],
) -> bool:
    """Return False for client / structured API errors (e.g. task not found)."""
    if 400 <= http_code < 500:
        return False
    if payload is not None:
        api_code = payload.get("code")
        if api_code is not None and int(api_code) != 200:
            return False
    if http_code >= 500:
        return http_code in (502, 503, 504)
    return True


def _raise_query_error_from_http(exc: urllib.error.HTTPError, body: str) -> None:
    payload, raw = _decode_query_error_body(body)
    raise RuntimeError(
        _format_query_api_error(http_code=exc.code, payload=payload, raw_body=raw or body)
    ) from exc


def _pick_task_id(resp: dict[str, Any]) -> str:
    block = _get_result_block(resp)
    for key in ("task_id", "taskId", "id", "job_id", "jobId"):
        val = block.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    raise KeyError(f"task_id not found in submit response: {resp}")


def _pick_status(resp: dict[str, Any]) -> Optional[str]:
    block = _get_result_block(resp)
    for key in ("status", "state", "phase", "job_status"):
        if key in block:
            return _normalize_status(block.get(key))
    return _normalize_status(resp.get("status"))


def _pick_progress(resp: dict[str, Any]) -> Optional[float]:
    block = _get_result_block(resp)
    progress = block.get("progress")
    if progress is None:
        return None
    try:
        return float(progress)
    except (TypeError, ValueError):
        return None


def _pick_error_code(resp: dict[str, Any]) -> Optional[int]:
    block = _get_result_block(resp)
    code = block.get("error_code")
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _pick_job_message(resp: dict[str, Any]) -> Optional[str]:
    block = _get_result_block(resp)
    for key in ("message", "error_message", "err_msg"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    top = resp.get("message")
    if isinstance(top, str) and top.strip():
        return top.strip()
    return None


def _is_http_url(value: str) -> bool:
    v = value.strip()
    return v.startswith("http://") or v.startswith("https://")


def _strip_shell_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


_RESULT_URL_JSON_KEYS = (
    "result_url",
    "resultUrl",
    "url",
    "bos_url",
    "bosUrl",
    "download_url",
)


def _decode_output_path_string(raw: str) -> str:
    """Normalize gateway ``output.path`` that may be shell-quoted JSON."""
    text = raw.strip()
    for _ in range(3):
        prev = text
        text = _strip_shell_quotes(text)
        if text == prev:
            break
    while text.endswith("')") or text.endswith('")'):
        text = text[:-2].rstrip()
    while text and text[-1] in ");":
        text = text[:-1].rstrip()
    text = text.replace("\\'", "'").replace('\\"', '"')
    if "\\n" in text or "\\r" in text or "\\t" in text:
        text = (
            text.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
        )
    return text.strip()


def _url_from_json_mapping(obj: dict[str, Any]) -> Optional[str]:
    for key in _RESULT_URL_JSON_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and _is_http_url(val):
            return val.strip()
    return None


def _resolve_download_url(value: Any) -> Optional[str]:
    """Extract https URL from direct URL, JSON object, or quoted JSON string."""
    if isinstance(value, dict):
        return _url_from_json_mapping(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if _is_http_url(text):
        return text

    decoded = _decode_output_path_string(text)
    if _is_http_url(decoded):
        return decoded

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        url = _url_from_json_mapping(parsed)
        if url:
            return url

    match = re.search(
        r'["\']?(?:result_url|resultUrl|bos_url)["\']?\s*:\s*["\'](https?://[^"\']+)',
        decoded,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)

    match = re.search(r"https?://[^\s'\"\\]+", decoded)
    if match:
        return match.group(0).rstrip("',\"})")
    return None


def _pick_result_url(resp: dict[str, Any]) -> Optional[str]:
    block = _get_result_block(resp)

    output = block.get("output")
    if isinstance(output, dict):
        for key in ("path", "url", "result_url", "bos_url"):
            url = _resolve_download_url(output.get(key))
            if url:
                return url
    else:
        url = _resolve_download_url(output)
        if url:
            return url

    for key in (*_RESULT_URL_JSON_KEYS, "path"):
        url = _resolve_download_url(block.get(key))
        if url:
            return url

    stack: list[Any] = [block]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for v in cur.values():
                url = _resolve_download_url(v)
                if url:
                    return url
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list, str)):
                    stack.append(v)
    return None


def _format_job_failure(resp: dict[str, Any]) -> str:
    status = _pick_status(resp)
    error_code = _pick_error_code(resp)
    message = _pick_job_message(resp)
    parts = [f"status={status!r}"]
    if error_code is not None:
        parts.append(f"error_code={error_code}")
    if message:
        parts.append(f"message={message}")
    return ", ".join(parts)


def _is_terminal_success(status: Optional[str], success_set: set[str]) -> bool:
    if status is None:
        return False
    return status.lower() in success_set


def _is_terminal_failure(status: Optional[str], failure_set: set[str]) -> bool:
    if status is None:
        return False
    return status.lower() in failure_set


def _result_filename(task_id: str) -> str:
    """Build local result filename from gateway task_id."""
    safe = re.sub(r"[^\w.-]+", "_", str(task_id).strip())
    if not safe:
        raise ValueError(f"invalid task_id for result filename: {task_id!r}")
    return f"{safe}.json"


def _redact_url(url: str) -> str:
    """Hide signed query strings from logs."""
    parsed = urllib.parse.urlparse(url)
    if parsed.query:
        return urllib.parse.urlunparse(parsed._replace(query="..."))
    return url


def _download_url(
    url: str,
    output_dir: Path,
    timeout: float,
    *,
    filename: str,
    attempts: int = 3,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    last_err: Optional[BaseException] = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, BrokenPipeError) as exc:
            last_err = exc
            print(
                f"[download] attempt {attempt}/{attempts} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < attempts:
                time.sleep(1.5 * attempt)

    assert last_err is not None
    raise last_err


def _load_result_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise TypeError(f"result file must be JSON object: {path}")
    return doc


def _extract_inference_result(doc: dict[str, Any]) -> dict[str, Any]:
    """Unwrap gateway result payload ``{\"result\": {...}}``."""
    if set(doc.keys()) <= {"result_url", "bos"} and doc.get("result_url"):
        raise ValueError(
            "downloaded JSON only contains result_url; "
            'expected {"result": {...}} inference payload'
        )
    inner = doc.get("result")
    if isinstance(inner, dict):
        if any(k in inner for k in ("text", "ppl", "logits", "values", "shape", "error")):
            if "error" in inner:
                raise RuntimeError(str(inner["error"]))
            return inner
        nested = inner.get("result")
        if isinstance(nested, dict):
            if "error" in nested:
                raise RuntimeError(str(nested["error"]))
            return nested
        return inner
    if "error" in doc:
        raise RuntimeError(str(doc["error"]))
    return doc


def _pick_embedding_shape(inference: dict[str, Any]) -> Optional[list[Any]]:
    shape = inference.get("shape")
    if shape is not None:
        return list(shape) if isinstance(shape, (list, tuple)) else [shape]

    emb = inference.get("embedding")
    if isinstance(emb, dict):
        inner_shape = emb.get("shape")
        if inner_shape is not None:
            return (
                list(inner_shape)
                if isinstance(inner_shape, (list, tuple))
                else [inner_shape]
            )
        values = emb.get("values")
    else:
        values = inference.get("values")

    if not isinstance(values, list) or not values:
        return None

    dims: list[int] = []
    level: Any = values
    while isinstance(level, list):
        dims.append(len(level))
        if not level:
            break
        level = level[0]
    return dims or None


def _pick_task_type_from_query(resp: dict[str, Any]) -> Optional[str]:
    """Read task_type from gateway query payload when present."""
    for container in (_get_result_block(resp), resp):
        params = container.get("parameters")
        if isinstance(params, dict):
            task_type = params.get("task_type")
            if task_type is not None and str(task_type).strip():
                return _normalize_task_type(task_type)
        task_type = container.get("task_type")
        if task_type is not None and str(task_type).strip():
            return _normalize_task_type(task_type)
    return None


def _infer_task_type_from_inference(inference: dict[str, Any]) -> Optional[str]:
    if inference.get("ppl") is not None or inference.get("logits") is not None:
        return "logits"
    if inference.get("values") is not None or inference.get("shape") is not None:
        return "embedding"
    if inference.get("text") is not None:
        return "generate"
    return None


def _resolve_summary_task_type(
    *,
    cli_task_type: Optional[str],
    query_resp: dict[str, Any],
    result_doc: dict[str, Any],
) -> str:
    if cli_task_type is not None and str(cli_task_type).strip():
        return _normalize_task_type(cli_task_type)
    from_query = _pick_task_type_from_query(query_resp)
    if from_query is not None:
        return from_query
    try:
        inference = _extract_inference_result(result_doc)
    except (ValueError, RuntimeError):
        inference = result_doc
    inferred = _infer_task_type_from_inference(inference)
    if inferred is not None:
        return inferred
    return "embedding"


def _print_result_summary(task_type: str, result_path: Path, doc: dict[str, Any]) -> None:
    abs_path = result_path.resolve()
    print(f"[done] saved: {abs_path}")

    try:
        inference = _extract_inference_result(doc)
    except (ValueError, RuntimeError) as exc:
        print(f"[done] could not parse inference result: {exc}", file=sys.stderr)
        return

    task = task_type.lower()
    if task == "generate":
        text = inference.get("text")
        if text is None:
            print("[generate] text is missing in result", file=sys.stderr)
        else:
            print(f"[generate] text:\n{text}")
        token_count = inference.get("generated_token_count")
        if token_count is not None:
            print(f"[generate] generated_token_count={token_count}")
        finish_reason = inference.get("finish_reason")
        if finish_reason is not None:
            print(f"[generate] finish_reason={finish_reason}")
    elif task == "logits":
        ppl = inference.get("ppl")
        if ppl is None:
            print("[logits] ppl=N/A")
        else:
            print(f"[logits] ppl={ppl}")
        loss = inference.get("loss")
        if loss is not None:
            print(f"[logits] loss={loss}")
    elif task == "embedding":
        shape = _pick_embedding_shape(inference)
        if shape is None:
            print("[embedding] shape=N/A", file=sys.stderr)
        else:
            print(f"[embedding] shape={shape}")
        values = inference.get("values")
        if values is None and isinstance(inference.get("embedding"), dict):
            values = inference["embedding"].get("values")
        if isinstance(values, list):
            print(f"[embedding] value_count={len(values)}")
        prefix_token_count = inference.get("prefix_token_count")
        if prefix_token_count is not None:
            print(f"[embedding] prefix_token_count={prefix_token_count}")


def _normalize_gateway_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.rstrip("/"))
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", "")
    )


def _build_query_url(task_id: str, *, gateway_url: str) -> str:
    base = _normalize_gateway_url(gateway_url)
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}task_id={urllib.parse.quote(task_id, safe='')}"


def _validate_cli_mode(args: argparse.Namespace) -> None:
    """Require submit fields or ``--task-id`` for history fetch."""
    task_id = getattr(args, "task_id", None)
    if task_id is not None and str(task_id).strip():
        return
    if not args.prompt and not args.prompt_file:
        raise ValueError("one of --prompt, --prompt-file, or --task-id is required")
    if not args.task_type:
        raise ValueError("--task-type is required when submitting a new job")


def _poll_job_until_done(
    task_id: str,
    *,
    gateway_url: str,
    token: str,
    http_timeout: float,
    poll_interval: float,
    timeout: float,
    success_set: set[str],
    failure_set: set[str],
    running_set: set[str],
    save_query_json: Optional[str] = None,
) -> dict[str, Any]:
    query_url = _build_query_url(task_id, gateway_url=gateway_url)
    print(f"[query] GET {gateway_url}?task_id=...")
    deadline = time.time() + max(1.0, timeout)
    last_resp: dict[str, Any] = {}

    while time.time() < deadline:
        print(f"[query] GET {query_url}")
        try:
            last_resp = _query_job(query_url, http_timeout, token=token)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if not _should_retry_query_http_error(exc.code, _decode_query_error_body(body)[0]):
                _raise_query_error_from_http(exc, body)
            print(f"[query] HTTP {exc.code}: {body}", file=sys.stderr)
            time.sleep(max(0.5, poll_interval))
            continue
        except urllib.error.URLError as exc:
            print(f"[query] network error: {exc}", file=sys.stderr)
            time.sleep(max(0.5, poll_interval))
            continue

        _ensure_api_ok(last_resp, context="query")

        status = _pick_status(last_resp)
        progress = _pick_progress(last_resp)
        progress_text = f" progress={progress}" if progress is not None else ""
        if _is_terminal_success(status, success_set):
            state_hint = "success"
        elif _is_terminal_failure(status, failure_set):
            state_hint = "failed"
        elif status in running_set:
            state_hint = "running"
        else:
            state_hint = "unknown"
        print(
            f"[query] status={status!r}{progress_text} "
            f"code={last_resp.get('code')} ({state_hint})"
        )

        if _is_terminal_failure(status, failure_set):
            raise RuntimeError(f"job failed: {_format_job_failure(last_resp)}")
        if _is_terminal_success(status, success_set):
            break
        time.sleep(max(0.5, poll_interval))
    else:
        raise TimeoutError(
            f"job did not succeed within {timeout}s, "
            f"last={_format_job_failure(last_resp)}"
        )

    if save_query_json:
        save_path = Path(save_query_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(last_resp, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[query] saved response: {save_path}")

    return last_resp


def _download_task_result(
    task_id: str,
    query_resp: dict[str, Any],
    *,
    output_dir: Path,
    http_timeout: float,
) -> Path:
    result_url = _pick_result_url(query_resp)
    if not result_url:
        block = _get_result_block(query_resp)
        output_path = ""
        output = block.get("output")
        if isinstance(output, dict):
            output_path = str(output.get("path") or "")
        raise KeyError(
            "download URL not found in query response "
            f"(expected result.output.path as http(s) URL or "
            f'{{"result_url": "..."}}, got path={output_path!r})'
        )

    print(f"[download] {_redact_url(result_url)}")
    result_filename = _result_filename(task_id)
    return _download_url(
        result_url,
        output_dir,
        timeout=max(http_timeout, 300.0),
        filename=result_filename,
    )


def _set_param(params: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        params[key] = value


def _validate_prompt_length(prompt: str, *, source: str) -> str:
    length = len(prompt)
    if length > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"prompt too long from {source}: {length} characters "
            f"(max {MAX_PROMPT_LENGTH}, model context {MAX_MODEL_LEN} "
            f"minus {RESERVED_PROMPT_TOKENS} reserved tokens)"
        )
    return prompt


def _sequence_from_prompt_text(text: str) -> str:
    """Keep raw DNA, or drop FASTA headers if the file starts with ``>``."""
    stripped = text.strip()
    if not stripped.startswith(">"):
        return stripped
    bases = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.startswith(">")
    ]
    return "".join(bases)


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        text = _sequence_from_prompt_text(text)
        if not text:
            raise ValueError(f"empty prompt file: {args.prompt_file}")
        return _validate_prompt_length(text, source=args.prompt_file)
    assert args.prompt is not None
    return _validate_prompt_length(args.prompt, source="--prompt")


def _wrap_generate_params_json_for_gateway(gen: dict[str, Any]) -> str:
    """Wrap compact JSON in single quotes for downstream shell command assembly."""
    if not gen:
        return DEFAULT_GENERATE_PARAMS_JSON_VALUE
    payload = json.dumps(gen, ensure_ascii=False, separators=(",", ":"))
    if not payload or payload == "{}":
        return DEFAULT_GENERATE_PARAMS_JSON_VALUE
    if "'" in payload:
        payload = payload.replace("'", "'\"'\"'")
    return f"'{payload}'"


_INFERENCE_TASK_TYPES = frozenset({"embedding", "generate", "logits"})


def _normalize_task_type(task_type: Any) -> str:
    return str(task_type).strip().lower()


def _assign_generate_params_json(
    parameters: dict[str, Any],
    args: argparse.Namespace,
    *,
    template_gen: Optional[dict[str, Any]] = None,
) -> None:
    """Ensure parameters.generate_params_json is always set for inference jobs."""
    task = _normalize_task_type(parameters.get("task_type", args.task_type))
    if task not in _INFERENCE_TASK_TYPES:
        parameters.pop("generate_params_json", None)
        return

    if task == "generate":
        base = template_gen if template_gen is not None else {}
        gen = _build_generate_params_dict(args, base)
        parameters["generate_params_json"] = _wrap_generate_params_json_for_gateway(gen)
    else:
        # embedding / logits: field required by gateway, default empty object
        parameters["generate_params_json"] = DEFAULT_GENERATE_PARAMS_JSON_VALUE


def _decode_generate_params_json_text(text: str) -> dict[str, Any]:
    text = _strip_shell_quotes(text.strip())
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError(
            f"generate_params_json must be a JSON object, got {type(parsed)!r}"
        )
    return parsed


def _parse_generate_params_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return _decode_generate_params_json_text(value)
    raise TypeError(f"generate_params_json has unsupported type: {type(value)!r}")


def _build_generate_params_dict(
    args: argparse.Namespace,
    base: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    gen: dict[str, Any] = dict(base or {})
    if args.generate_params_json:
        try:
            extra = json.loads(args.generate_params_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --generate-params-json: {exc}") from exc
        if not isinstance(extra, dict):
            raise ValueError("--generate-params-json must decode to a JSON object")
        gen.update(extra)
    _set_param(gen, "max_tokens", args.max_tokens)
    _set_param(gen, "min_tokens", args.min_tokens)
    _set_param(gen, "top_p", args.top_p)
    _set_param(gen, "top_k", args.top_k)
    _set_param(gen, "min_p", args.min_p)
    _set_param(gen, "temperature", args.temperature)
    _set_param(gen, "repetition_penalty", args.repetition_penalty)
    _set_param(gen, "presence_penalty", args.presence_penalty)
    _set_param(gen, "frequency_penalty", args.frequency_penalty)
    return gen


def _species_arg_value(args: argparse.Namespace) -> Optional[str]:
    """The user-facing species scientific name, or ``None`` when not given."""
    name = getattr(args, "species", None)
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def _resolved_species_raw(args: argparse.Namespace) -> Optional[str]:
    """Resolve the requested species to the internal gateway species id.

    ``--species`` (scientific name) is the user-facing input; it is resolved
    through the client-side ``species.py`` lookup.  ``--species-raw`` remains
    only as a hidden, backward-compatible escape hatch for callers that already
    know the id.
    """
    name = _species_arg_value(args)
    if name is not None:
        return to_species_raw(name)

    raw = getattr(args, "species_raw", None)
    if raw is None or species_raw_is_unset(raw):
        return None
    return str(raw).strip()


def _log_species_submit_mode(args: argparse.Namespace) -> None:
    """Remind operator that no species was requested ([spMASK] on worker)."""
    if _species_arg_value(args) is not None:
        return
    print("[species] no --species given; job runs with no specific species")


def _species_raw_for_submit(args: argparse.Namespace) -> str:
    """Internal species id for parameters.species_raw, or SPECIES_RAW_EMPTY for [spMASK]."""
    species_raw = _resolved_species_raw(args)
    if species_raw is None:
        return SPECIES_RAW_EMPTY
    return species_raw


def _apply_species_parameters(
    params: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    Merge the species into gateway ``parameters``.

    Always sets ``parameters.species_raw`` (string): the internal species id
    resolved from ``--species`` (or ``--species-raw``), or ``SPECIES_RAW_EMPTY``
    when no species is requested.  Worker receives ``--species-raw`` with the
    same value (platform maps this field only).
    """
    params.pop("species_token_id", None)
    params.pop("species_token", None)
    params["species_raw"] = _species_raw_for_submit(args)


def _user_facing_parameters(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Display copy of ``parameters`` for the submit log.

    Keeps the internal ``species_raw`` id out of user-facing logs and shows the
    requested scientific name under ``species`` instead.
    """
    params = payload.get("parameters")
    if not isinstance(params, dict):
        return {}
    preview = dict(params)
    name = _species_arg_value(args)
    if name is not None:
        preview["species"] = name
    preview.pop("species_raw", None)
    return preview


def _build_parameters_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble payload.parameters from CLI flags (species finalized in merge)."""
    params: dict[str, Any] = {
        "prompt": _resolve_prompt(args),
        "task_type": args.task_type,
        "species_raw": SPECIES_RAW_EMPTY,
    }
    _set_param(params, "request_id", args.request_id)
    return params


def _merge_parameters_into_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    base = payload.get("parameters")
    if base is None:
        merged: dict[str, Any] = {}
    elif isinstance(base, dict):
        merged = dict(base)
    else:
        raise TypeError(f"payload.parameters must be object, got {type(base)!r}")

    template_gen = _parse_generate_params_json(merged.get("generate_params_json"))
    for key in _GENERATE_FLAT_KEYS:
        if key in merged and key not in template_gen:
            template_gen[key] = merged[key]

    merged.update(_build_parameters_from_cli(args))

    _assign_generate_params_json(merged, args, template_gen=template_gen)

    for key in _GENERATE_FLAT_KEYS:
        merged.pop(key, None)

    _apply_species_parameters(merged, args)

    payload["parameters"] = merged


def _ensure_generate_params_json_before_submit(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Last-chance guard: inference jobs must include generate_params_json."""
    parameters = payload.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError(f"payload.parameters must be object, got {type(parameters)!r}")

    task = _normalize_task_type(parameters.get("task_type", args.task_type))
    if task not in _INFERENCE_TASK_TYPES:
        parameters.pop("generate_params_json", None)
        return

    raw = parameters.get("generate_params_json")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        parameters["generate_params_json"] = DEFAULT_GENERATE_PARAMS_JSON_VALUE
        return

    if isinstance(raw, dict) and not raw:
        parameters["generate_params_json"] = DEFAULT_GENERATE_PARAMS_JSON_VALUE
        return

    if task in ("embedding", "logits") and raw != DEFAULT_GENERATE_PARAMS_JSON_VALUE:
        # Non-generate tasks always send the default placeholder.
        parameters["generate_params_json"] = DEFAULT_GENERATE_PARAMS_JSON_VALUE


def _add_parameter_arguments(parser: argparse.ArgumentParser) -> None:
    core = parser.add_argument_group(
        "parameters (required for new jobs; omit when using --task-id)"
    )
    prompt_src = core.add_mutually_exclusive_group(required=False)
    prompt_src.add_argument(
        "--prompt",
        type=str,
        help=(
            f"DNA prompt sequence (max {MAX_PROMPT_LENGTH} chars; "
            f"{RESERVED_PROMPT_TOKENS} tokens reserved from {MAX_MODEL_LEN})."
        ),
    )
    prompt_src.add_argument(
        "--prompt-file",
        type=str,
        help=(
            f"Read prompt from file (max {MAX_PROMPT_LENGTH} chars; "
            f"{RESERVED_PROMPT_TOKENS} tokens reserved from {MAX_MODEL_LEN})."
        ),
    )
    core.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=("embedding", "generate", "logits"),
        help="Inference task type (required for submit; optional hint when using --task-id).",
    )
    core.add_argument(
        "--species",
        type=str,
        default=None,
        help=(
            "Species scientific name (e.g. \"Homo sapiens\"); resolved by the "
            "client to the internal species id used for inference. If omitted, "
            "the job runs without a specific species."
        ),
    )
    core.add_argument(
        "--species-raw",
        type=str,
        default=SPECIES_RAW_EMPTY,
        help=argparse.SUPPRESS,  # legacy / internal escape hatch for raw species ids
    )
    core.add_argument("--request-id", type=str, default=None, help="Optional request id.")

    gen = parser.add_argument_group("generate parameters (task_type=generate)")
    gen.add_argument(
        "--generate-params-json",
        type=str,
        default=None,
        help=(
            "Generate params as JSON object; merged into parameters.generate_params_json. "
            'Example: \'{"max_tokens":64,"top_k":1}\'. '
            "Flat flags below override keys from this JSON."
        ),
    )
    gen.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate.")
    gen.add_argument("--min-tokens", type=int, default=None, help="Min tokens to generate.")
    gen.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p.")
    gen.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    gen.add_argument("--min-p", type=float, default=None, help="Min-p sampling threshold.")
    gen.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 1.0).")
    gen.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (>=1.0; >1.0 discourages repeats).",
    )
    gen.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty (OpenAI-style).",
    )
    gen.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty (OpenAI-style).",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DNA async job submit/query/download CLI")
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=os.environ.get("INFERENCE_GATEWAY_URL", DEFAULT_GATEWAY_URL),
        help=f"Inference gateway URL (default: {DEFAULT_GATEWAY_URL}).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=f"API token for Authorization: Bearer header (or env {DEFAULT_TOKEN_ENV}).",
    )
    parser.add_argument(
        "--submit-file",
        type=str,
        default=DEFAULT_SUBMIT_FILE,
        help="Submit request body template (modelname, namespace, parameters, ...).",
    )
    _add_parameter_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to save downloaded result file ({task_id}.json).",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument(
        "--success-status",
        type=str,
        default=",".join(DEFAULT_SUCCESS_STATUSES),
        help="Comma-separated terminal success statuses.",
    )
    parser.add_argument(
        "--failure-status",
        type=str,
        default=",".join(DEFAULT_FAILURE_STATUSES),
        help="Comma-separated terminal failure statuses.",
    )
    parser.add_argument(
        "--save-submit-json",
        type=str,
        default=None,
        help="Optional path to save submit response JSON.",
    )
    parser.add_argument(
        "--save-query-json",
        type=str,
        default=None,
        help="Optional path to save last query response JSON.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help=(
            "Existing gateway task id: skip submit, poll until success, "
            "download {task_id}.json (history / retry download)."
        ),
    )
    args = parser.parse_args()

    try:
        _validate_cli_mode(args)
    except ValueError as exc:
        parser.error(str(exc))

    gateway_url = _normalize_gateway_url(args.gateway_url)
    token = _resolve_bearer_token(args)
    success_set = {s.strip().lower() for s in args.success_status.split(",") if s.strip()}
    failure_set = {s.strip().lower() for s in args.failure_status.split(",") if s.strip()}
    running_set = {s.strip().lower() for s in DEFAULT_RUNNING_STATUSES}

    if args.task_id is not None and str(args.task_id).strip():
        task_id = str(args.task_id).strip()
        print(f"[fetch] task_id={task_id}")
        last_resp = _poll_job_until_done(
            task_id,
            gateway_url=gateway_url,
            token=token,
            http_timeout=args.http_timeout,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            success_set=success_set,
            failure_set=failure_set,
            running_set=running_set,
            save_query_json=args.save_query_json,
        )
        out_path = _download_task_result(
            task_id,
            last_resp,
            output_dir=Path(args.output_dir),
            http_timeout=args.http_timeout,
        )
        result_doc = _load_result_json(out_path)
        summary_type = _resolve_summary_task_type(
            cli_task_type=args.task_type,
            query_resp=last_resp,
            result_doc=result_doc,
        )
        _print_result_summary(summary_type, out_path, result_doc)
        return 0

    root = _repo_root()
    submit_arg = Path(args.submit_file)
    if submit_arg.is_absolute() or submit_arg.is_file():
        submit_path = submit_arg
    else:
        submit_path = root / submit_arg
    submit_text = _load_text(submit_path)
    payload = _extract_json_object(submit_text, submit_path)

    _merge_parameters_into_payload(payload, args)
    _ensure_generate_params_json_before_submit(payload, args)
    _log_species_submit_mode(args)

    params_preview = _user_facing_parameters(payload, args)
    print(
        f"[submit] POST {gateway_url} "
        f"parameters={json.dumps(params_preview, ensure_ascii=False)}"
    )
    try:
        submit_resp = _submit_job(
            gateway_url,
            payload,
            args.http_timeout,
            token=token,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"submit failed: HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"submit failed: {exc}") from exc

    _ensure_api_ok(submit_resp, context="submit")
    if args.save_submit_json:
        save_path = Path(args.save_submit_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(submit_resp, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[submit] saved response: {save_path}")

    task_id = _pick_task_id(submit_resp)
    print(f"[submit] code={submit_resp.get('code')} task_id={task_id}")

    last_resp = _poll_job_until_done(
        task_id,
        gateway_url=gateway_url,
        token=token,
        http_timeout=args.http_timeout,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        success_set=success_set,
        failure_set=failure_set,
        running_set=running_set,
        save_query_json=args.save_query_json,
    )
    out_path = _download_task_result(
        task_id,
        last_resp,
        output_dir=Path(args.output_dir),
        http_timeout=args.http_timeout,
    )
    result_doc = _load_result_json(out_path)
    _print_result_summary(args.task_type, out_path, result_doc)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpeciesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
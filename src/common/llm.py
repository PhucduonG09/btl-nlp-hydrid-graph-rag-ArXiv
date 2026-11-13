"""Wrapper gọi LLM qua API OpenAI-compatible: retry, cache theo hash, ghi usage và served_model.

Đọc cấu hình từ .env ở thư mục gốc repo (biến môi trường có sẵn được ưu tiên).
"""
import os, json, time, hashlib, random, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:      # python-dotenv chưa cài: dùng biến môi trường của hệ thống
    pass

CACHE_PATH = ROOT / "data" / "cache" / "llm.jsonl"
_lock = threading.Lock()
_cache = None
_client = None
_mock = None
_abort = threading.Event()
_NO_TEMP = set()          # model từ chối temperature tùy chỉnh (vd. một số model Claude/o-series)
_consec_fatal = 0          # số lỗi nghiêm trọng liên tiếp (reset khi có một lần gọi thành công)


class FatalLLMError(RuntimeError):
    """Lỗi không nên thử lại (sai key, sai tên model, hết tiền...)."""


def set_mock(fn):
    """Dùng cho test: fn(model, messages, kw) -> (text, tokens_in, tokens_out)."""
    global _mock
    _mock = fn


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise FatalLLMError("Chưa có LLM_API_KEY: mở file .env ở thư mục gốc repo và dán key sau dấu '='.")
        _client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=key, timeout=float(os.environ.get("LLM_TIMEOUT", "120")), max_retries=0,
                         default_headers={"Accept-Encoding":"identity"})
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        _cache[d["key"]] = d
                    except Exception:
                        continue          # dòng hỏng do bị ngắt giữa chừng
    return _cache


def chat(messages, model=None, *, temperature=0.0, max_tokens=None, json_mode=False,
         run_id=0, use_cache=True):
    """Trả (text, usage). usage = {in, out, latency_s, served_model, cached}."""
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise FatalLLMError("Chưa đặt LLM_MODEL trong .env")
    kw = {"temperature": temperature}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    key = hashlib.sha256(json.dumps([model, messages, kw, run_id], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
    cache = _load_cache()
    if use_cache and key in cache:
        d = cache[key]
        return d["text"], {**d["usage"], "cached": True}
    if _abort.is_set():
        raise FatalLLMError("Đã dừng do lỗi nghiêm trọng trước đó.")

    t0 = time.time()
    if _mock is not None:
        text, tin, tout = _mock(model, messages, kw)
        usage = {"in": tin, "out": tout, "served_model": model}
    else:
        text, usage = _call_with_retry(model, messages, kw)
    usage["latency_s"] = round(time.time() - t0, 2)
    usage["cached"] = False
    if use_cache:
        rec = {"key": key, "text": text, "usage": {k: v for k, v in usage.items() if k != "cached"}}
        with _lock:
            cache[key] = rec
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return text, usage


def _call_with_retry(model, messages, kw):
    global _consec_fatal
    from openai import RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
    client = _get_client()
    kw = dict(kw)
    if model in _NO_TEMP:
        kw.pop("temperature", None)
    for attempt in range(7):
        try:
            r = client.chat.completions.create(model=model, messages=messages, **kw)
            text = (r.choices[0].message.content or "") if r.choices else ""
            u = getattr(r, "usage", None)
            _consec_fatal = 0
            return text, {"in": getattr(u, "prompt_tokens", 0), "out": getattr(u, "completion_tokens", 0),
                          "served_model": getattr(r, "model", None)}
        except (RateLimitError, APIConnectionError, APITimeoutError):
            pass
        except APIStatusError as e:
            if e.status_code == 400 and "temperature" in str(e).lower() and "temperature" in kw:
                _NO_TEMP.add(model)          # model không nhận temperature tùy chỉnh: bỏ đi, dùng mặc định rồi gọi lại
                kw.pop("temperature")
                continue
            transient = e.status_code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)   # 52x: lỗi/timeout ở cổng trung gian
            # Cổng trung gian đôi khi trả 400 kèm thông báo "hệ thống đang bận / retry shortly": đó là lỗi tạm thời, không phải lỗi yêu cầu
            if e.status_code == 400 and any(k in str(e).lower() for k in ("busy", "retry shortly", "try again", "đang bận", "overloaded")):
                transient = True
            # Một số nhà cung cấp trung gian thỉnh thoảng trả 401/403 thoáng qua (kênh upstream lỗi): thử lại vài lần
            flaky_auth = e.status_code in (401, 403) and attempt < 4
            if not (transient or flaky_auth):
                _consec_fatal += 1
                if _consec_fatal >= 6:            # chỉ dừng toàn bộ khi lỗi liên tiếp, không phải một lần lẻ
                    _abort.set()
                raise FatalLLMError(f"HTTP {e.status_code}: {str(e)[:300]}") from e
        time.sleep(min(60, 2 * 2 ** attempt) + random.random())
    raise RuntimeError("Hết lượt thử lại (rate limit hoặc mạng).")


def run_parallel(fn, items, workers=6, desc=""):
    """Chạy fn(item) song song, giữ thứ tự. Trả (results, errors). Dừng sớm nếu gặp lỗi nghiêm trọng."""
    results, errors = [None] * len(items), []

    def wrap(i):
        try:
            results[i] = fn(items[i])
        except Exception as e:                       # noqa: BLE001
            errors.append((i, repr(e)))
    try:
        from tqdm import tqdm
        bar = lambda it: tqdm(it, total=len(items), desc=desc)
    except ImportError:
        bar = lambda it: it
    with ThreadPoolExecutor(workers) as ex:
        list(bar(ex.map(wrap, range(len(items)))))
    return results, errors


def cost_summary(usages):
    """Tổng token tính tiền (bỏ qua lần trúng cache) và ước tính USD nếu có LLM_PRICE_IN/OUT."""
    tin = sum(u["in"] for u in usages if u and not u.get("cached"))
    tout = sum(u["out"] for u in usages if u and not u.get("cached"))
    pin, pout = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
    usd = (tin * float(pin) + tout * float(pout)) / 1e6 if pin and pout else None
    return {"tokens_in": tin, "tokens_out": tout, "usd_est": usd,
            "served_models": sorted({u.get("served_model") or "?" for u in usages if u})}

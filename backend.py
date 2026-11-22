"""
Backend for the RAG + tools chatbot (LangGraph).

Highlights
- Per-session PDF index (no more global state shared between users/tabs)
- System prompt + conversation memory (MemorySaver) + history trimming
- 10 tools: web search, PDF RAG, calculator, time, text analyzer,
  weather, unit converter, currency converter, date calculator, webpage reader
"""
import ast
import ipaddress
import math
import operator
import os
import re
import socket
import threading
from collections import Counter
from datetime import date, datetime, timedelta, timezone as dt_timezone
from typing import Annotated, Optional, TypedDict
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain Core
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# LangChain Providers / community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional OpenAI provider
try:
    from langchain_openai import ChatOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ─── Config ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPEN_API_KEY"] = OPENAI_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Asia/Karachi")

MAX_HISTORY_MESSAGES = 30  # messages sent to the LLM per call
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGChatbot/2.0)"}

# ─── Model Selection ───────────────────────────────────────────────
# Temperature is kept low: tool calling is more reliable than at 0.7.
if HAS_OPENAI and OPENAI_API_KEY:
    model = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)
    ACTIVE_PROVIDER = f"OpenAI ({OPENAI_MODEL})"
elif GROQ_API_KEY:
    model = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    ACTIVE_PROVIDER = f"Groq ({GROQ_MODEL})"
else:
    raise EnvironmentError(
        "No API key found. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
    )

# ─── Embeddings (lazy: app starts fast, model loads on first PDF) ──
_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# ─── Per-session RAG store ─────────────────────────────────────────
# session_id (= LangGraph thread_id) -> {"retriever", "name", "pages", "chunks"}
_RAG_STORES: dict = {}
_RAG_LOCK = threading.Lock()


def load_and_index_pdf(pdf_path: str, session_id: str = "default", display_name: Optional[str] = None):
    """Load a PDF, chunk it, build a FAISS index and attach it to this session."""
    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise ValueError("The PDF has no pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No extractable text found (is this a scanned PDF?).")

    store = FAISS.from_documents(chunks, get_embeddings())
    retriever = store.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}
    )
    with _RAG_LOCK:
        _RAG_STORES[session_id] = {
            "retriever": retriever,
            "name": display_name or os.path.basename(pdf_path),
            "pages": len(docs),
            "chunks": len(chunks),
        }
    return len(docs), len(chunks)


def clear_pdf(session_id: str = "default") -> None:
    with _RAG_LOCK:
        _RAG_STORES.pop(session_id, None)


def get_pdf_info(session_id: str = "default") -> Optional[dict]:
    entry = _RAG_STORES.get(session_id)
    if not entry:
        return None
    return {k: entry[k] for k in ("name", "pages", "chunks")}


# ─── State ─────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── Helpers ───────────────────────────────────────────────────────
def _get_tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        raise ValueError(
            f"Unknown timezone '{name}'. Use an IANA name like 'Asia/Karachi' or 'UTC'."
        )


def _today() -> date:
    try:
        return datetime.now(_get_tz(DEFAULT_TIMEZONE)).date()
    except ValueError:  # tzdata missing on this machine
        return datetime.now(dt_timezone.utc).date()


def _get_json(url: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _is_public_url(url: str) -> bool:
    """Allow only http(s) URLs that resolve to public IPs (basic SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


# ─── Tools ─────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchResults(
    api_wrapper=DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=5),
    num_results=5,
)

# --- calculator (safe expression evaluator, no eval()) ---
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
}
_MATH_CONSTS = {"pi": math.pi, "e": math.e}


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > 10_000:
        raise ValueError("Result is too large")
    return value


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id in _MATH_CONSTS:
        return _MATH_CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _MATH_FUNCS
        and not node.keywords
    ):
        return _MATH_FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> dict:
    """
    Evaluate a math expression exactly. Use this for ANY arithmetic.
    Supports + - * / // % ** and parentheses, constants pi and e, and the functions
    sqrt, log, log10, exp, sin, cos, tan, floor, ceil, abs, round.
    Example: "(1250 * 1.18) / 4" or "sqrt(144) + 2**10"
    """
    try:
        if len(expression) > 200:
            return {"error": "Expression is too long"}
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero is not allowed"}
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return {"error": f"Could not evaluate expression: {exc}"}


# --- RAG ---
@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Search the PDF the user uploaded and return the most relevant passages with page numbers.
    Use this whenever the user asks about the uploaded document.
    """
    session_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    entry = _RAG_STORES.get(session_id)
    if entry is None:
        return {
            "query": query,
            "context": [],
            "message": "No PDF is loaded yet. Ask the user to upload a document in the sidebar first.",
        }

    results = entry["retriever"].invoke(query)
    if not results:
        return {"query": query, "context": [], "message": "No relevant content found in the document."}

    return {
        "query": query,
        "document": entry["name"],
        "context": [
            {"page": doc.metadata.get("page", 0) + 1, "text": doc.page_content}
            for doc in results
        ],
    }


# --- time ---
@tool
def get_current_time(timezone: str = DEFAULT_TIMEZONE, fmt: str = "%Y-%m-%d %H:%M:%S") -> dict:
    """
    Get the current date and time in an IANA timezone (e.g. 'Asia/Karachi', 'UTC',
    'America/New_York', 'Europe/Berlin'). Optional fmt is a Python strftime format.
    """
    try:
        now = datetime.now(_get_tz(timezone))
        return {
            "current_time": now.strftime(fmt),
            "timezone": timezone,
            "weekday": now.strftime("%A"),
            "iso": now.isoformat(),
        }
    except ValueError as exc:
        return {"error": str(exc)}


# --- text analyzer ---
_STOPWORDS = set(
    "a an the and or but if of to in on at for with from by as is are was were be been it this that "
    "these those i you he she we they my your his her our their not no so than then there which who "
    "will would can could should have has had do does did".split()
)


@tool
def text_analyzer(text: str, analysis_type: str = "all") -> dict:
    """
    Analyze text and return statistics.
    analysis_type: all, word_count, character_count, sentence_count, average_word_length,
    reading_time_minutes, top_keywords
    """
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = len(words)

    tokens = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in _STOPWORDS and len(w) > 2]
    stats = {
        "word_count": word_count,
        "character_count": len(text),
        "sentence_count": len(sentences),
        "average_word_length": round(sum(len(w) for w in words) / word_count, 2) if word_count else 0,
        "reading_time_minutes": round(word_count / 200, 2),  # ~200 wpm
        "top_keywords": [w for w, _ in Counter(tokens).most_common(5)],
    }

    aliases = {"char_count": "character_count", "avg_word_length": "average_word_length"}
    key = aliases.get(analysis_type, analysis_type)
    if key in stats:
        return {key: stats[key]}
    return stats


# --- weather (Open-Meteo, no API key) ---
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    56: "Freezing drizzle", 57: "Dense freezing drizzle", 61: "Light rain", 63: "Rain",
    65: "Heavy rain", 66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow",
    73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light rain showers",
    81: "Rain showers", 82: "Violent rain showers", 85: "Snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}


@tool
def get_weather(city: str) -> dict:
    """Get the current weather and a 3-day forecast for a city, e.g. 'Lahore'. Temperatures in °C."""
    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city, "count": 1},
        )
        places = geo.get("results") or []
        if not places:
            return {"error": f"City '{city}' was not found"}
        place = places[0]

        data = _get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        cur, daily = data["current"], data["daily"]
        return {
            "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "current": {
                "temperature_c": cur["temperature_2m"],
                "feels_like_c": cur["apparent_temperature"],
                "humidity_pct": cur["relative_humidity_2m"],
                "wind_kmh": cur["wind_speed_10m"],
                "precipitation_mm": cur["precipitation"],
                "conditions": _WMO_CODES.get(cur["weather_code"], "Unknown"),
            },
            "forecast": [
                {
                    "date": daily["time"][i],
                    "max_c": daily["temperature_2m_max"][i],
                    "min_c": daily["temperature_2m_min"][i],
                    "rain_chance_pct": daily["precipitation_probability_max"][i],
                }
                for i in range(len(daily["time"]))
            ],
        }
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"error": f"Weather lookup failed: {exc}"}


# --- unit converter ---
_UNITS = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
    "mass": {"mg": 1e-6, "g": 1e-3, "kg": 1, "lb": 0.45359237, "oz": 0.028349523125},
    "volume": {"ml": 1e-3, "l": 1, "tsp": 0.00492892159375, "tbsp": 0.01478676478125, "cup": 0.2365882365, "gal": 3.785411784},
    "time": {"s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800},
    "data": {"byte": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4},
}
_UNIT_ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km", "kilometers": "km",
    "centimeter": "cm", "centimeters": "cm", "millimeter": "mm", "mile": "mi", "miles": "mi",
    "foot": "ft", "feet": "ft", "inch": "in", "inches": "in", "yard": "yd", "yards": "yd",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "pound": "lb", "pounds": "lb",
    "lbs": "lb", "ounce": "oz", "ounces": "oz", "liter": "l", "liters": "l", "litre": "l",
    "litres": "l", "milliliter": "ml", "gallon": "gal", "gallons": "gal", "cups": "cup",
    "second": "s", "seconds": "s", "sec": "s", "minute": "min", "minutes": "min", "hour": "h",
    "hours": "h", "hr": "h", "days": "day", "weeks": "week", "bytes": "byte", "b": "byte",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}
_TEMP_TO_C = {"c": lambda v: v, "f": lambda v: (v - 32) * 5 / 9, "k": lambda v: v - 273.15}
_TEMP_FROM_C = {"c": lambda v: v, "f": lambda v: v * 9 / 5 + 32, "k": lambda v: v + 273.15}


def _norm_unit(unit: str) -> str:
    unit = unit.strip().lower().replace("°", "").replace(".", "")
    return _UNIT_ALIASES.get(unit, unit)


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert between units of length (mm, cm, m, km, in, ft, yd, mi), mass (mg, g, kg, lb, oz),
    volume (ml, l, tsp, tbsp, cup, gal - US), time (s, min, h, day, week),
    temperature (c, f, k) and data size (byte, kb, mb, gb, tb, kib, mib, gib, tib).
    """
    src, dst = _norm_unit(from_unit), _norm_unit(to_unit)

    if src in _TEMP_TO_C and dst in _TEMP_FROM_C:
        result = _TEMP_FROM_C[dst](_TEMP_TO_C[src](value))
    else:
        for table in _UNITS.values():
            if src in table and dst in table:
                result = value * table[src] / table[dst]
                break
        else:
            return {"error": f"Cannot convert '{from_unit}' to '{to_unit}' (unknown or incompatible units)"}

    return {"value": value, "from": src, "to": dst, "result": round(result, 6)}


# --- currency converter (open.er-api.com, no API key) ---
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert an amount between currencies using live exchange rates.
    Use 3-letter ISO codes, e.g. USD, PKR, EUR, GBP, AED.
    """
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    if not (re.fullmatch(r"[A-Z]{3}", src) and re.fullmatch(r"[A-Z]{3}", dst)):
        return {"error": "Use 3-letter currency codes such as USD or PKR"}
    try:
        data = _get_json(f"https://open.er-api.com/v6/latest/{src}")
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"Exchange-rate lookup failed: {exc}"}

    rate = (data.get("rates") or {}).get(dst)
    if data.get("result") != "success" or rate is None:
        return {"error": f"No exchange rate available for {src} -> {dst}"}
    return {
        "amount": amount,
        "from": src,
        "to": dst,
        "rate": rate,
        "converted": round(amount * rate, 4),
        "rates_last_updated": data.get("time_last_update_utc"),
        "source": "open.er-api.com",
    }


# --- date calculator ---
def _parse_date(value: Optional[str]) -> date:
    value = (value or "").strip().lower()
    if value in ("", "today", "now"):
        return _today()
    return datetime.strptime(value, "%Y-%m-%d").date()


@tool
def date_calculator(start_date: str = "today", end_date: Optional[str] = None, days_to_add: int = 0) -> dict:
    """
    Date arithmetic. Dates are YYYY-MM-DD ('today' is allowed).
    - With end_date: returns the number of days between start_date and end_date.
    - Without end_date: returns start_date shifted by days_to_add (negative = in the past).
    """
    try:
        start = _parse_date(start_date)
        if end_date:
            end = _parse_date(end_date)
            days = (end - start).days
            return {"start": start.isoformat(), "end": end.isoformat(), "days_between": days, "weeks": round(days / 7, 2)}
        result = start + timedelta(days=days_to_add)
        return {
            "start": start.isoformat(),
            "days_added": days_to_add,
            "result_date": result.isoformat(),
            "weekday": result.strftime("%A"),
        }
    except ValueError:
        return {"error": "Dates must be in YYYY-MM-DD format"}


# --- webpage reader ---
@tool
def read_webpage(url: str, max_chars: int = 4000) -> dict:
    """
    Fetch a public web page and return its readable text (max_chars between 500 and 8000).
    Use it to open a link the user gives you or a result from web search.
    """
    max_chars = max(500, min(int(max_chars), 8000))
    try:
        current = url.strip()
        resp = None
        for _ in range(4):  # follow up to 3 redirects, re-checking each hop
            if not _is_public_url(current):
                return {"error": "Only public http(s) URLs that resolve to a public address can be opened"}
            resp = requests.get(
                current, headers=HTTP_HEADERS, timeout=10, stream=True, allow_redirects=False
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                current = urljoin(current, resp.headers.get("Location", ""))
                resp.close()
                resp = None
                continue
            break
        if resp is None:
            return {"error": "Too many redirects"}

        resp.raise_for_status()
        if "html" not in resp.headers.get("Content-Type", "").lower() and "text" not in resp.headers.get("Content-Type", "").lower():
            resp.close()
            return {"error": "The URL does not point to a text/HTML page"}

        raw = resp.raw.read(1_000_000, decode_content=True)  # cap download at ~1 MB
        resp.close()

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return {
            "url": current,
            "title": title,
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        }
    except requests.RequestException as exc:
        return {"error": f"Could not fetch the page: {exc}"}


# ─── Bind all tools to model ──────────────────────────────────────
tools = [
    search_tool,
    calculator,
    rag_tool,
    get_current_time,
    text_analyzer,
    get_weather,
    convert_units,
    convert_currency,
    date_calculator,
    read_webpage,
]
llm_with_tools = model.bind_tools(tools)


# ─── System prompt ────────────────────────────────────────────────
def build_system_prompt() -> str:
    try:
        now = datetime.now(_get_tz(DEFAULT_TIMEZONE))
    except ValueError:
        now = datetime.now(dt_timezone.utc)
    return (
        "You are a helpful, accurate assistant inside a RAG chatbot.\n"
        f"Current date/time: {now:%A, %d %B %Y, %H:%M} ({DEFAULT_TIMEZONE}).\n\n"
        "Tool rules:\n"
        "- Questions about the user's uploaded PDF: call rag_tool and answer only from the returned "
        "context, citing page numbers like (p. 3). If the context does not contain the answer, say so.\n"
        "- Any arithmetic: calculator. Unit or currency conversion: convert_units / convert_currency. "
        "Date maths: date_calculator. Time in a city or timezone: get_current_time. Weather: get_weather.\n"
        "- Current events, prices, or facts you are unsure about: use web search; use read_webpage to "
        "open a specific URL from the results or one the user provides.\n"
        "- Do not call tools for greetings or for general knowledge you are certain about.\n"
        "- Never invent tool results. If a tool returns an error, tell the user briefly and suggest what to try.\n"
        "Keep answers concise and professional."
    )


# ─── Graph Nodes ──────────────────────────────────────────────────
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # counts messages, not tokens
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",
        include_system=False,
    )
    if not history:  # safety net: never send an empty/invalid window
        history = state["messages"]

    prompt = [SystemMessage(content=build_system_prompt())] + list(history)

    # Llama models on Groq occasionally emit a malformed tool call; one retry usually fixes it.
    for attempt in range(2):
        try:
            return {"messages": [llm_with_tools.invoke(prompt)]}
        except Exception as exc:
            if attempt == 1 or "tool_use_failed" not in str(exc):
                raise


tool_node = ToolNode(tools)

# ─── Build & Compile Graph ────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# MemorySaver keeps per-thread history (thread_id = Streamlit session). It lives in RAM:
# fine for a demo; swap for SqliteSaver/PostgresSaver if you deploy it.
Workflow = graph.compile(checkpointer=MemorySaver())

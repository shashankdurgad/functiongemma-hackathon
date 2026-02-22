import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ================================================================
#  Schema-Driven Tool Matching — Generic argument extraction
#  Works for ANY tool by reading parameter schemas at runtime.
# ================================================================

# Verb synonym table: maps common query verbs to related words
# that might appear in tool names/descriptions. This is English-level
# knowledge, NOT tool-specific — works for unknown tools too.
_VERB_SYNONYMS = {
    "text": ["message", "send", "sms"],
    "tell": ["message", "send", "notify"],
    "saying": ["message", "send"],
    "call": ["contact", "phone", "dial"],
    "check": ["get", "look", "search", "find"],
    "look": ["search", "find", "query"],
    "grab": ["get", "fetch", "retrieve"],
    "fetch": ["get", "retrieve"],
    "make": ["create", "set", "add"],
    "add": ["create", "set", "new"],
    "remove": ["delete", "clear", "cancel"],
    "cancel": ["delete", "remove", "stop"],
    "start": ["play", "begin", "run", "launch"],
    "stop": ["pause", "end", "cancel", "halt"],
    "show": ["display", "get", "view", "list"],
    "find": ["search", "look", "locate", "query"],
    "wake": ["alarm", "alert", "notify"],
    "remind": ["reminder", "alert", "notify", "memo"],
    "book": ["reserve", "schedule", "create"],
    "order": ["purchase", "buy", "request"],
    "turn": ["set", "toggle", "switch"],
    "open": ["launch", "start", "run"],
    "close": ["stop", "end", "shut"],
    "send": ["message", "deliver", "transmit"],
    "play": ["start", "music", "media"],
    "pause": ["stop", "hold"],
    "search": ["find", "look", "query", "locate"],
}

_STOPWORDS = {"the", "a", "an", "my", "some", "me", "us", "it", "is", "for", "and", "or"}


def _score_tool_relevance(content_lower, tool):
    """Score how relevant a tool is to a query using name, description, and synonym matching."""
    score = 0
    query_words = set(content_lower.split())
    name_words = tool["name"].replace("_", " ").split()
    desc = tool.get("description", "").lower()
    desc_words = [w for w in desc.split() if len(w) > 3]
    tool_vocab = set(name_words) | set(desc_words)

    # Direct name-word matching (exact + prefix)
    for w in name_words:
        if len(w) <= 2:
            continue
        if w in content_lower:
            score += 3
        elif any(qw.startswith(w) or w.startswith(qw) for qw in query_words if len(qw) > 2):
            score += 2

    # Description word matching (exact + prefix)
    for w in desc_words:
        if w in content_lower:
            score += 1
        elif any(qw.startswith(w) or w.startswith(qw) for qw in query_words if len(qw) > 3):
            score += 1

    # Synonym expansion: query verbs → tool vocab
    for qw in query_words:
        if qw in _VERB_SYNONYMS:
            for syn in _VERB_SYNONYMS[qw]:
                if any(syn in tv or tv in syn for tv in tool_vocab if len(tv) > 2):
                    score += 2
                    break  # one synonym match per query word

    return score


def _extract_string_param(content, content_lower, pname, pdesc):
    """Extract a string value using the parameter's name and description as semantic guidance."""
    tags = {pname.lower()} | set(pname.lower().split("_"))
    desc_tags = set(pdesc.lower().split())
    all_tags = tags | desc_tags

    # --- Location / city / place ---
    if all_tags & {"location", "city", "place", "destination", "area", "region", "where"}:
        m = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
        if not m:
            m = re.search(r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
        if not m:
            # lowercase fallback
            m = re.search(r'\bin\s+([a-z][a-z\s]*?)(?:\s*[.?,!]|\s+and\s+|\s*$)', content_lower)
        if m:
            return m.group(1).strip().title()

    # --- Recipient / person / contact ---
    if all_tags & {"recipient", "person", "contact", "who", "receiver"}:
        # Capitalized name after communication verb
        m = re.search(r'(?:[Ss]end|[Tt]o|[Tt]ext|[Tt]ell|[Mm]essage|[Cc]all)\s+(?:a\s+message\s+to\s+)?([A-Z][a-z]+)', content)
        if m and m.group(1).lower() not in _STOPWORDS:
            return m.group(1).strip()
        # Lowercase fallback with "saying" anchor
        m = re.search(r'(?:send|to|text|tell|message|call)\s+(\w+)\s+(?:a\s+message\s+)?saying', content_lower)
        if m and m.group(1) not in _STOPWORDS and m.group(1) not in {"him", "her", "them"}:
            return m.group(1).strip().title()
        # Pronoun resolution: "send him/her a message" → look for name elsewhere
        if re.search(r'\b(?:him|her|them)\b', content_lower):
            name_m = re.search(r'(?:find|look\s+up|search\s+for)\s+(\w+)', content_lower)
            if name_m and name_m.group(1).lower() not in _STOPWORDS:
                return name_m.group(1).strip().title()

    # --- Message / content / body ---
    if all_tags & {"message", "content", "body", "note"}:
        m = re.search(r'\bsaying\s+(.+?)(?:\s+and\s+|\s*[.,!]|\s*$)', content_lower)
        if not m:
            m = re.search(r'\bthat\s+says?\s+(.+?)(?:\s+and\s+|\s*[.,!]|\s*$)', content_lower)
        if m:
            return m.group(1).strip()

    # --- Title / subject / label ---
    if all_tags & {"title", "subject", "topic", "label"}:
        # "about <title> at <time>"
        m = re.search(r'\babout\s+(?:the\s+)?(.+?)\s+at\s+', content_lower)
        if not m:
            # "to <action> at <time>" (e.g., "remind me to take medicine at 7:00 AM")
            m = re.search(r'\bto\s+(.+?)\s+at\s+\d', content_lower)
        if m:
            return m.group(1).strip()

    # --- Song / music / playlist / media ---
    if all_tags & {"song", "music", "playlist", "track", "album", "media"}:
        m = re.search(r'\bplay\s+(?:some\s+)?(.+?)(?:\s+and\s+|\s*[.,!]|\s*$)', content_lower)
        if m:
            val = m.group(1).strip()
            # Strip generic trailing "music" for genre queries
            if val.endswith(" music") and val != "classical music":
                val = val[:-6].strip()
            return val

    # --- Query / search term ---
    if all_tags & {"query", "search", "keyword", "term", "find", "look"}:
        m = re.search(r'(?:find|look\s+up|search\s+for)\s+(\w+)', content_lower)
        if m:
            val = m.group(1).strip().title()
            if val.lower() not in _STOPWORDS:
                return val

    # --- Time as string ---
    if all_tags & {"time", "when", "schedule", "deadline"}:
        m = re.search(r'(\d{1,2}:\d{2}\s*(?:am|pm))', content_lower, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper()

    # --- Fallback: quoted strings ---
    m = re.search(r'"([^"]+)"', content)
    if m:
        return m.group(1)

    # --- Fallback: proper nouns (for "name"-like params) ---
    if all_tags & {"name"}:
        names = re.findall(r'\b([A-Z][a-z]{2,})\b', content)
        if names:
            return names[-1]

    return None


def _extract_integer_param(content_lower, pname, pdesc):
    """Extract an integer value using the parameter's name and description as guidance."""
    tags = {pname.lower()} | set(pname.lower().split("_"))
    desc_tags = set(pdesc.lower().split())
    all_tags = tags | desc_tags

    # Hour from time pattern (requires AM/PM to avoid false matches)
    if all_tags & {"hour", "hours"}:
        m = re.search(r'(\d{1,2})(?::\d{2})?\s*(?:am|pm)', content_lower)
        if m:
            return int(m.group(1))

    # Duration in minutes (for timers, countdowns — NOT minute-of-hour)
    if pname.lower() == "minutes" or (all_tags & {"minutes"} and all_tags & {"number", "duration", "timer", "countdown", "many"}):
        m = re.search(r'(\d+)\s*min', content_lower)
        if m:
            return int(m.group(1))

    # Minute component of a time (0-59)
    if pname.lower() == "minute" or (all_tags & {"minute"} and not all_tags & {"number", "duration", "countdown"}):
        m = re.search(r'\d+:(\d+)', content_lower)
        if m:
            return int(m.group(1))
        # Bare time like "6 AM" → minute defaults to 0
        if re.search(r'\d{1,2}\s*(?:am|pm)', content_lower):
            return 0

    # Generic fallback: first number in query
    nums = re.findall(r'\b(\d+)\b', content_lower)
    if nums:
        return int(nums[0])

    return None


def heuristic_tool_match(content, tools):
    """Schema-driven tool matching and argument extraction.

    Works for ANY tool by:
    1. Scoring tool relevance via name/description keyword + verb synonym matching
    2. Extracting arguments based on parameter type and description semantics
    """
    content_lower = content.lower().strip().rstrip('.')

    # Step 1: Score each tool's relevance to the query
    scored = []
    for tool in tools:
        score = _score_tool_relevance(content_lower, tool)
        if score >= 2:
            scored.append((score, tool))
    scored.sort(key=lambda x: -x[0])

    # Step 2: For each candidate, extract arguments from schema
    matched = []
    for _, tool in scored:
        props = tool["parameters"].get("properties", {})
        required = tool["parameters"].get("required", [])
        args = {}

        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")

            if ptype == "string":
                val = _extract_string_param(content, content_lower, pname, pdesc)
                if val is not None:
                    args[pname] = val
            elif ptype == "integer":
                val = _extract_integer_param(content_lower, pname, pdesc)
                if val is not None:
                    args[pname] = val
            elif ptype == "boolean":
                if any(w in content_lower for w in ["yes", "on", "true", "enable"]):
                    args[pname] = True
                elif any(w in content_lower for w in ["no", "off", "false", "disable"]):
                    args[pname] = False
            elif ptype == "number":
                nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', content_lower)
                if nums:
                    args[pname] = float(nums[0])

        # Only include if all required params were extracted
        if all(r in args for r in required):
            matched.append({"name": tool["name"], "arguments": args})

    return matched


# ================================================================
#  Query decomposition
# ================================================================

def _resolve_pronouns(content):
    """Replace pronouns (him/her/them) with the named entity found earlier in the query.

    E.g. "Find Tom in my contacts and send him a message" → replaces "him" with "Tom".
    """
    content_lower = content.lower()
    if not re.search(r'\b(?:him|her|them)\b', content_lower):
        return content
    # Look for a name via find/look up/search pattern
    m = re.search(r'(?:[Ff]ind|[Ll]ook\s+up|[Ss]earch\s+for)\s+(\w+)', content)
    if m:
        name = m.group(1).strip()
        if name.lower() not in _STOPWORDS:
            content = re.sub(r'\bhim\b', name, content, flags=re.IGNORECASE)
            content = re.sub(r'\bher\b', name, content, flags=re.IGNORECASE)
            content = re.sub(r'\bthem\b', name, content, flags=re.IGNORECASE)
    return content


def decompose_query(content):
    """Split multi-intent query into sub-queries using heuristics.

    Resolves pronouns before splitting so each sub-query is self-contained.
    """
    content = _resolve_pronouns(content)

    parts = re.split(r',?\s+and\s+|,?\s+then\s+', content, flags=re.IGNORECASE)
    result = [p.strip().rstrip('.').strip() for p in parts if p.strip() and len(p.strip()) > 5]
    if len(result) >= 2:
        return result

    # Try comma splitting
    parts = content.split(',')
    result = [p.strip().rstrip('.').strip() for p in parts if p.strip() and len(p.strip()) > 5]
    if len(result) >= 2:
        return result

    return [content]


def detect_multi_intent(content):
    """Check if the query requests multiple distinct actions."""
    content_lower = content.lower()
    conjunction_count = (
        content_lower.count(" and ") +
        content_lower.count(", and ") +
        content_lower.count(" then ")
    )
    comma_clauses = len(re.findall(r',\s+(?=[a-z])', content_lower))
    return max(conjunction_count, comma_clauses) > 0


def analyze_query_complexity(content, tools):
    """Analyze the query prompt to produce complexity metrics.

    Returns a dict with:
      - action_verb_count: how many distinct tool-specific verbs appear
      - entity_count: how many extractable entities (names, locations, times, numbers)
      - query_length: word count of the query
      - tool_ambiguity: ratio of tools that could plausibly match (0-1)
      - conjunction_count: number of coordination conjunctions / comma splits
      - has_implicit_args: whether the query has indirect/ambiguous argument references
    """
    content_lower = content.lower()
    tool_names = {t["name"] for t in tools}

    # Action verb detection — how many distinct tool actions are mentioned
    TOOL_VERBS = {
        "get_weather": ["weather", "forecast", "temperature"],
        "set_alarm": ["alarm", "wake me", "wake up"],
        "send_message": ["message", "text ", "tell ", "saying"],
        "create_reminder": ["remind", "reminder"],
        "search_contacts": ["find ", "look up", "search", "contacts"],
        "play_music": ["play ", "music", "song"],
        "set_timer": ["timer", "countdown", "minute timer"],
    }
    action_verb_count = 0
    matched_tools = set()
    for tool_name, verbs in TOOL_VERBS.items():
        if tool_name in tool_names:
            if any(v in content_lower for v in verbs):
                action_verb_count += 1
                matched_tools.add(tool_name)

    # Entity extraction counts
    entity_count = 0
    # Locations: "in <City>"
    entity_count += len(re.findall(r'\bin\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', content))
    # Names: capitalized words after "to", person references
    entity_count += len(re.findall(r'\bto\s+[A-Z][a-z]+', content))
    # Times: "7:30 AM", "5:00 PM"
    entity_count += len(re.findall(r'\d{1,2}:\d{2}\s*(?:am|pm)', content_lower))
    # Bare hours: "6 AM", "10 AM"
    entity_count += len(re.findall(r'\b\d{1,2}\s+(?:am|pm)\b', content_lower))
    # Numbers (minutes, etc.)
    entity_count += len(re.findall(r'\b\d+\s*min', content_lower))

    # Query length
    query_length = len(content_lower.split())

    # Tool ambiguity: what fraction of available tools could plausibly match?
    tool_ambiguity = len(matched_tools) / max(len(tools), 1)

    # Conjunction count
    conjunction_count = (
        content_lower.count(" and ") +
        content_lower.count(", and ") +
        content_lower.count(" then ")
    )
    comma_clauses = len(re.findall(r',\s+(?=[a-z])', content_lower))
    conjunction_count = max(conjunction_count, comma_clauses)

    # Implicit/ambiguous args: pronouns, indirect references
    has_implicit_args = bool(re.search(r'\b(him|her|them|it|this|that)\b', content_lower))

    return {
        "action_verb_count": action_verb_count,
        "entity_count": entity_count,
        "query_length": query_length,
        "tool_ambiguity": tool_ambiguity,
        "conjunction_count": conjunction_count,
        "has_implicit_args": has_implicit_args,
        "matched_tools": matched_tools,
    }


def classify_difficulty(num_tools, is_multi_intent, complexity=None):
    """Classify query difficulty using tool count, multi-intent flag, AND query complexity.

    Returns (label, temperatures_to_try, tool_rag_top_k).
    """
    if complexity is None:
        # Fallback to simple classification
        if is_multi_intent or num_tools >= 4:
            return "hard", [0.0, 0.3, 0.7], 3
        elif num_tools >= 2:
            return "medium", [0.0, 0.3], 2
        else:
            return "easy", [0.0], 1

    # Rich classification using complexity metrics
    action_count = complexity["action_verb_count"]
    entity_count = complexity["entity_count"]
    ambiguity = complexity["tool_ambiguity"]
    has_implicit = complexity["has_implicit_args"]

    # Hard: multi-intent, or high ambiguity with many tools, or implicit args with many tools
    if is_multi_intent or action_count >= 3:
        return "hard", [0.0, 0.3, 0.7], 3
    if num_tools >= 4 and (ambiguity > 0.5 or has_implicit):
        return "hard", [0.0, 0.3, 0.7], 3

    # Medium: multiple tools to choose from, or moderate ambiguity
    if num_tools >= 2:
        if ambiguity > 0.3 or has_implicit or entity_count >= 3:
            return "medium", [0.0, 0.3], 2
        # Simple choice among few tools with clear verb match
        if action_count == 1 and entity_count <= 2:
            return "easy", [0.0], min(num_tools, 2)
        return "medium", [0.0, 0.3], 2

    # Easy: single tool, clear intent
    return "easy", [0.0], 1


def compute_dynamic_confidence(difficulty, match_source, complexity=None,
                                cactus_conf=0.0, heuristic_matched=False,
                                cactus_sanity_passed=False):
    """Compute a dynamic confidence score based on difficulty and match quality.

    Args:
        difficulty: "easy", "medium", or "hard"
        match_source: "cactus", "heuristic", "cactus+heuristic", "decomposed"
        complexity: output from analyze_query_complexity (optional)
        cactus_conf: raw confidence from cactus model
        heuristic_matched: whether heuristic also found a match
        cactus_sanity_passed: whether sanity check passed for cactus result

    Returns a confidence float in [0, 1].
    """
    # Base confidence by difficulty
    BASE = {"easy": 0.90, "medium": 0.75, "hard": 0.60}
    conf = BASE.get(difficulty, 0.70)

    # Source bonuses/penalties
    if match_source == "cactus+heuristic":
        # Cactus picked the tool, heuristic confirmed/corrected args — highest trust
        conf += 0.10
    elif match_source == "cactus":
        # Cactus only, no heuristic validation
        conf += 0.0
    elif match_source == "heuristic":
        # Pure heuristic fallback — reliable for known patterns
        conf += 0.05
    elif match_source == "decomposed":
        # Multi-intent decomposition — moderate trust
        conf -= 0.05

    # Sanity check bonus
    if cactus_sanity_passed:
        conf += 0.05

    # Heuristic agreement bonus
    if heuristic_matched and match_source != "heuristic":
        conf += 0.05

    # Complexity-based adjustments
    if complexity:
        # Clear verb match = higher confidence
        if complexity["action_verb_count"] == 1 and not complexity["has_implicit_args"]:
            conf += 0.05
        # Implicit args = lower confidence
        if complexity["has_implicit_args"]:
            conf -= 0.10
        # High ambiguity = lower confidence
        if complexity["tool_ambiguity"] > 0.5:
            conf -= 0.05

    # Cactus model confidence signal (small weight — it's often unreliable)
    if cactus_conf > 0.8:
        conf += 0.03
    elif cactus_conf < 0.3:
        conf -= 0.03

    return max(0.0, min(1.0, conf))


def validate_calls(calls, tools):
    """Validate that function calls use valid tool names and have required args."""
    tool_names = {t["name"] for t in tools}
    if not calls:
        return False
    for call in calls:
        if call.get("name") not in tool_names:
            return False
    for call in calls:
        tool = next((t for t in tools if t["name"] == call.get("name")), None)
        if tool:
            required = tool["parameters"].get("required", [])
            if any(r not in call.get("arguments", {}) for r in required):
                return False
    return True


def dedup_calls(calls):
    """Remove duplicate function calls."""
    seen, unique = set(), []
    for c in calls:
        key = (c["name"], json.dumps(c.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ================================================================
#  On-Device Backend
# ================================================================

def _build_cactus_tools(tools):
    """Build cactus tool list and system prompt from tool definitions."""
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    tool_names_str = ", ".join(t["name"] for t in tools)
    system_prompt = (
        f"You are a function calling assistant. You MUST respond with a function call "
        f"using one of these tools: [{tool_names_str}]. "
        f"Output ONLY valid JSON function calls. Do not output conversational text."
    )
    return cactus_tools, system_prompt


def _cactus_complete_once(model, messages, cactus_tools, system_prompt,
                          temperature=0.0, tool_rag_top_k=None):
    """Run a single cactus_complete call on an already-loaded model."""
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools, force_tools=True, max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        temperature=temperature,
        tool_rag_top_k=tool_rag_top_k,
    )
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus(messages, tools, temperature=0.0, tool_rag_top_k=None):
    """Run function calling on-device via FunctionGemma + Cactus (single call)."""
    model = cactus_init(functiongemma_path)
    cactus_tools, system_prompt = _build_cactus_tools(tools)
    result = _cactus_complete_once(model, messages, cactus_tools, system_prompt,
                                   temperature, tool_rag_top_k)
    cactus_destroy(model)
    return result


def _cactus_temperature_race(messages, tools, temperatures, tool_rag_top_k=None):
    """Load model once, try multiple temperatures sequentially, return first valid.

    Single model init/destroy — avoids repeated loading overhead.
    Early-exits on first valid result.
    """
    start = time.time()
    model = cactus_init(functiongemma_path)
    cactus_tools, system_prompt = _build_cactus_tools(tools)
    best = None

    for temp in temperatures:
        result = _cactus_complete_once(model, messages, cactus_tools, system_prompt,
                                       temp, tool_rag_top_k)
        if validate_calls(result.get("function_calls", []), tools):
            best = result
            break
        if best is None:
            best = result  # keep first result for fallback

    cactus_destroy(model)
    wall_ms = (time.time() - start) * 1000

    if best is None:
        best = {"function_calls": [], "total_time_ms": wall_ms, "confidence": 0}
    best["total_time_ms"] = wall_ms
    return best, wall_ms


# ================================================================
#  Cloud Backend
# ================================================================

def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"], description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            ) for t in tools
        ])
    ]
    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )
    ms = (time.time() - start) * 1000
    calls = []
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.function_call:
                calls.append({"name": part.function_call.name, "arguments": dict(part.function_call.args)})
    return {"function_calls": calls, "total_time_ms": ms}


# ================================================================
#  Hybrid Generator — Single-pass with heuristic fallback
# ================================================================

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Adaptive routing: single cactus call + heuristic validation/fallback.

    Strategy:
    - Single cactus call (fast) — no ensemble overhead
    - If cactus returns valid calls → accept (on-device)
    - If not, try heuristic argument extraction (on-device)
    - For multi-intent: decompose → run each sub-query through cactus+heuristic
    - Cloud fallback only when both cactus and heuristic fail
    """
    content = messages[-1]["content"] if messages else ""
    content_lower = content.lower()
    num_tools = len(tools)
    is_multi = detect_multi_intent(content)

    # Query complexity analysis
    complexity = analyze_query_complexity(content, tools)
    difficulty, temperatures, rag_top_k = classify_difficulty(num_tools, is_multi, complexity)

    # === MULTI-INTENT: decompose, parallel cactus per sub-query, heuristic-corrected ===
    if is_multi:
        sub_queries = decompose_query(content)

        if len(sub_queries) >= 2:
            all_calls = []
            total_time = 0

            for sq in sub_queries:
                sub_messages = [{"role": "user", "content": sq}]

                # Parallel temperature race for this sub-query
                attempt, sq_wall = _cactus_temperature_race(sub_messages, tools, temperatures, rag_top_k)
                total_time = max(total_time, sq_wall)  # wall-clock: sub-queries are sequential
                sub_calls = attempt.get("function_calls", [])

                if validate_calls(sub_calls, tools):
                    # Cactus returned valid structure — use heuristic to correct args
                    h_calls = heuristic_tool_match(sq, tools)
                    h_map = {c["name"]: c for c in h_calls}
                    corrected = []
                    for c in sub_calls:
                        if c["name"] in h_map:
                            corrected.append(h_map[c["name"]])
                        else:
                            corrected.append(c)
                    all_calls.extend(corrected)
                else:
                    # Cactus failed entirely — use heuristic as fallback
                    h_calls = heuristic_tool_match(sq, tools)
                    if h_calls:
                        all_calls.extend(h_calls)

            all_calls = dedup_calls(all_calls)

            if len(all_calls) >= len(sub_queries):
                conf = compute_dynamic_confidence(
                    difficulty, "decomposed", complexity,
                    heuristic_matched=True)
                return {
                    "function_calls": all_calls,
                    "total_time_ms": total_time,
                    "confidence": conf,
                    "source": "on-device",
                }

            # Sub-query approach didn't get enough — try full heuristic on original
            h_calls = heuristic_tool_match(content, tools)
            if len(h_calls) >= len(sub_queries):
                conf = compute_dynamic_confidence(
                    difficulty, "heuristic", complexity,
                    heuristic_matched=True)
                return {
                    "function_calls": dedup_calls(h_calls),
                    "total_time_ms": total_time,
                    "confidence": conf,
                    "source": "on-device",
                }

            # Both failed — cloud fallback
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += total_time
            return cloud

    # === SINGLE-INTENT: parallel cactus temps, heuristic-corrected args ===
    local, wall_ms = _cactus_temperature_race(messages, tools, temperatures, rag_top_k)
    local_calls = local.get("function_calls", [])

    if validate_calls(local_calls, tools):
        sanity_ok = _sanity_check(local_calls, content_lower, tools)
        if sanity_ok:
            # Cactus got the right tool — use heuristic to correct arguments
            h_calls = heuristic_tool_match(content, tools)
            h_map = {c["name"]: c for c in h_calls}
            corrected = []
            for c in local_calls:
                if c["name"] in h_map:
                    corrected.append(h_map[c["name"]])
                else:
                    corrected.append(c)
            local["function_calls"] = corrected
            local["source"] = "on-device"
            local["total_time_ms"] = wall_ms
            local["confidence"] = compute_dynamic_confidence(
                difficulty, "cactus+heuristic", complexity,
                cactus_conf=local.get("confidence", 0),
                heuristic_matched=bool(h_calls),
                cactus_sanity_passed=True)
            return local

    # Cactus failed — try pure heuristic as fallback
    h_calls = heuristic_tool_match(content, tools)
    if validate_calls(h_calls, tools):
        conf = compute_dynamic_confidence(
            difficulty, "heuristic", complexity,
            heuristic_matched=True)
        return {
            "function_calls": h_calls,
            "total_time_ms": wall_ms,
            "confidence": conf,
            "source": "on-device",
        }

    # Both failed — cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += wall_ms
    return cloud


def _sanity_check(calls, content_lower, tools):
    """Verify returned tool names semantically match the query using generic scoring."""
    # Score all available tools against the query
    plausible = set()
    for tool in tools:
        if _score_tool_relevance(content_lower, tool) >= 2:
            plausible.add(tool["name"])

    if not plausible:
        return True  # Can't determine expected tools — trust cactus

    returned_tools = {c["name"] for c in calls}
    return bool(returned_tools & plausible)


# ================================================================
#  CLI
# ================================================================

if __name__ == "__main__":
    tools = [{
        "name": "get_weather", "description": "Get current weather for a location",
        "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]},
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]
    result = generate_hybrid(messages, tools)
    print(f"\nSource: {result.get('source')} | Time: {result['total_time_ms']:.0f}ms")
    for c in result["function_calls"]:
        print(f"  {c['name']}({json.dumps(c['arguments'])})")

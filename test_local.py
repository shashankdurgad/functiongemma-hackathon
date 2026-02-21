"""
Local testing without Cactus/Mac - mocks FunctionGemma responses to test routing logic.
Run: python test_local.py
"""

import json, random, re
from routing import make_hybrid_generator, decompose_query


# ============== Mock Backends ==============

def parse_request(content, tools):
    """Keyword-based parsing to simulate FunctionGemma tool calls."""
    calls, tool_map = [], {t["name"]: t for t in tools}
    content = content.lower()

    if "weather" in content and "get_weather" in tool_map:
        loc = next((c.title() for c in ["san francisco","london","paris","tokyo","berlin",
                    "new york","miami","chicago","seattle"] if c in content), "unknown")
        calls.append({"name": "get_weather", "arguments": {"location": loc}})

    if ("alarm" in content or "wake" in content) and "set_alarm" in tool_map:
        m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(?:am|AM)?', content)
        calls.append({"name": "set_alarm", "arguments": {
            "hour": int(m.group(1)) if m else 8, "minute": int(m.group(2) or 0) if m else 0}})

    if ("message" in content or "text" in content or "send" in content) and "send_message" in tool_map:
        recip = next((n.title() for n in ["alice","bob","john","dave","emma","lisa","tom","jake","sarah"] if n in content), "unknown")
        msg = "hello"
        if "saying" in content:
            msg = content[content.find("saying")+7:].strip().rstrip(".").split(" and ")[0].strip()
        calls.append({"name": "send_message", "arguments": {"recipient": recip, "message": msg}})

    if "timer" in content and "set_timer" in tool_map:
        m = re.search(r'(\d+)\s*(?:minute|min)', content)
        calls.append({"name": "set_timer", "arguments": {"minutes": int(m.group(1)) if m else 5}})

    if ("play" in content or "music" in content) and "play_music" in tool_map:
        song = next((s for kw, s in [("bohemian","Bohemian Rhapsody"),("jazz","jazz"),
            ("lo-fi","lo-fi beats"),("lofi","lo-fi beats"),("classical","classical music"),
            ("summer","summer hits")] if kw in content), "music")
        calls.append({"name": "play_music", "arguments": {"song": song}})

    if "remind" in content and "create_reminder" in tool_map:
        title = next((t for kw, t in [("meeting","meeting"),("medicine","take medicine"),
            ("groceries","groceries"),("dentist","call the dentist"),("stretch","stretch")] if kw in content), "reminder")
        tm = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', content)
        calls.append({"name": "create_reminder", "arguments": {"title": title, "time": tm.group(1).upper() if tm else "3:00 PM"}})

    if ("find" in content or "look up" in content or "search" in content) and "search_contacts" in tool_map:
        q = next((n.title() for n in ["bob","tom","sarah","jake"] if n in content), "unknown")
        calls.append({"name": "search_contacts", "arguments": {"query": q}})

    return calls


def mock_generate_cactus(messages, tools):
    """Simulate FunctionGemma with realistic success rates."""
    content = messages[-1]["content"].lower() if messages else ""
    num_tools, has_conj = len(tools), " and " in content
    diff = "easy" if num_tools == 1 and not has_conj else ("medium" if num_tools <= 3 and not has_conj else "hard")

    rates = {"easy": 0.9, "medium": 0.7, "hard": 0.4}
    ranges = {"easy": (0.6, 0.95), "medium": (0.4, 0.85), "hard": (0.2, 0.7)}
    success = random.random() < rates[diff]
    confidence = random.uniform(*ranges[diff])

    calls = parse_request(content, tools) if success else []
    if calls and random.random() < 0.1:
        if random.random() < 0.5:
            args = calls[0].get("arguments", {})
            if args: del args[list(args.keys())[0]]
        else:
            calls[0]["name"] = "hallucinated_tool"

    return {"function_calls": calls, "total_time_ms": random.uniform(30, 80), "confidence": confidence}


def mock_generate_cloud(messages, tools):
    """Mock cloud response - high accuracy."""
    content = messages[-1]["content"].lower() if messages else ""
    return {"function_calls": parse_request(content, tools), "total_time_ms": random.uniform(200, 400)}


def mock_cloud_decompose(content, tools):
    """Enhanced regex splitting for mock cloud decomposition."""
    parts = re.split(r',?\s+and\s+|,?\s+then\s+|,\s*and\s+|,\s+(?=[a-z])', content, flags=re.IGNORECASE)
    parts = [p.strip().rstrip('.').strip() for p in parts if p.strip() and len(p.strip()) > 5]
    if len(parts) >= 2:
        print(f"  Mock cloud decomposition: {parts}")
        return parts
    return decompose_query(content)


# ============== Wire Up ==============

generate_hybrid = make_hybrid_generator(mock_generate_cactus, mock_generate_cloud, mock_cloud_decompose)


# ============== Benchmark Runner ==============

if __name__ == "__main__":
    T = lambda name, desc, props, req: {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": req}}

    TOOLS = {
        "weather": T("get_weather", "Get current weather", {"location": "string"}, ["location"]),
        "alarm":   T("set_alarm", "Set an alarm", {"hour": "integer", "minute": "integer"}, ["hour", "minute"]),
        "msg":     T("send_message", "Send a message", {"recipient": "string", "message": "string"}, ["recipient", "message"]),
        "remind":  T("create_reminder", "Create a reminder", {"title": "string", "time": "string"}, ["title", "time"]),
        "search":  T("search_contacts", "Search contacts", {"query": "string"}, ["query"]),
        "music":   T("play_music", "Play music", {"song": "string"}, ["song"]),
        "timer":   T("set_timer", "Set a timer", {"minutes": "integer"}, ["minutes"]),
    }
    W, A, M, R, S, P, Ti = [TOOLS[k] for k in ["weather","alarm","msg","remind","search","music","timer"]]

    BENCHMARKS = [
        ("weather_sf",        "easy",   "What is the weather in San Francisco?",   [W],         [{"name":"get_weather","arguments":{"location":"San Francisco"}}]),
        ("alarm_10am",        "easy",   "Set an alarm for 10 AM.",                 [A],         [{"name":"set_alarm","arguments":{"hour":10,"minute":0}}]),
        ("message_alice",     "easy",   "Send a message to Alice saying good morning.", [M],    [{"name":"send_message","arguments":{"recipient":"Alice","message":"good morning"}}]),
        ("weather_london",    "easy",   "What's the weather like in London?",      [W],         [{"name":"get_weather","arguments":{"location":"London"}}]),
        ("alarm_6am",         "easy",   "Wake me up at 6 AM.",                     [A],         [{"name":"set_alarm","arguments":{"hour":6,"minute":0}}]),
        ("msg_among_three",   "medium", "Send a message to John saying hello.",    [W,M,A],     [{"name":"send_message","arguments":{"recipient":"John","message":"hello"}}]),
        ("weather_among_two", "medium", "What's the weather in Tokyo?",            [W,M],       [{"name":"get_weather","arguments":{"location":"Tokyo"}}]),
        ("alarm_among_three", "medium", "Set an alarm for 8:15 AM.",               [M,A,W],     [{"name":"set_alarm","arguments":{"hour":8,"minute":15}}]),
        ("music_among_three", "medium", "Play some jazz music.",                   [A,P,W],     [{"name":"play_music","arguments":{"song":"jazz"}}]),
        ("timer_among_three", "medium", "Set a timer for 10 minutes.",             [A,Ti,P],    [{"name":"set_timer","arguments":{"minutes":10}}]),
        ("msg_and_weather",   "hard",   "Send a message to Bob saying hi and get the weather in London.", [W,M,A],
            [{"name":"send_message","arguments":{"recipient":"Bob","message":"hi"}}, {"name":"get_weather","arguments":{"location":"London"}}]),
        ("alarm_and_weather", "hard",   "Set an alarm for 7:30 AM and check the weather in New York.", [W,A,M],
            [{"name":"set_alarm","arguments":{"hour":7,"minute":30}}, {"name":"get_weather","arguments":{"location":"New York"}}]),
        ("timer_and_music",   "hard",   "Set a timer for 20 minutes and play lo-fi beats.", [Ti,P,W,A],
            [{"name":"set_timer","arguments":{"minutes":20}}, {"name":"play_music","arguments":{"song":"lo-fi beats"}}]),
        ("weather_and_music", "hard",   "Check the weather in Miami and play summer hits.", [W,P,Ti,M],
            [{"name":"get_weather","arguments":{"location":"Miami"}}, {"name":"play_music","arguments":{"song":"summer hits"}}]),
    ]

    def _norm(v): return v.strip().lower() if isinstance(v, str) else v
    def _match(p, e):
        if p["name"] != e["name"]: return False
        return all(_norm(p.get("arguments",{}).get(k)) == _norm(v) for k, v in e.get("arguments",{}).items())
    def compute_f1(pred, exp):
        if not pred and not exp: return 1.0
        if not pred or not exp: return 0.0
        used, matched = set(), 0
        for e in exp:
            for i, p in enumerate(pred):
                if i not in used and _match(p, e): matched += 1; used.add(i); break
        pr, re_ = matched/len(pred), matched/len(exp)
        return 2*pr*re_/(pr+re_) if pr+re_ else 0.0

    print("=" * 60)
    print("  LOCAL MOCK TESTING")
    print("=" * 60)
    print(f"\nRunning {len(BENCHMARKS)} benchmark cases...\n")

    results = []
    for i, (name, diff, query, tools, expected) in enumerate(BENCHMARKS, 1):
        result = generate_hybrid([{"role": "user", "content": query}], tools)
        f1 = compute_f1(result["function_calls"], expected)
        src = result.get("source", "unknown")
        print(f"[{i:2}/{len(BENCHMARKS)}] {name:<24} | {diff:<6} | F1={f1:.2f} | {src}")
        results.append({"name": name, "difficulty": diff, "f1": f1, "source": src})

    print("\n" + "=" * 60 + "\n  SUMMARY\n" + "=" * 60)
    for d in ["easy", "medium", "hard"]:
        g = [r for r in results if r["difficulty"] == d]
        if g:
            print(f"  {d:<8}: F1={sum(r['f1'] for r in g)/len(g):.2f}, on-device={sum(1 for r in g if r['source']=='on-device')}/{len(g)}")
    on = sum(1 for r in results if r["source"] == "on-device")
    print(f"\n  Overall: F1={sum(r['f1'] for r in results)/len(results):.2f}, on-device={on}/{len(results)} ({100*on//len(results)}%)")
    print("=" * 60)

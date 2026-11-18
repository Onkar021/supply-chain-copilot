"""Agent 0 - Disruption Parser (LLM for language understanding, rules as safety net)."""
import json, os, re, unicodedata
from dataclasses import dataclass, field, asdict
import pandas as pd

UF = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}
STATE_NAMES = {"sao paulo": "SP", "rio de janeiro": "RJ", "minas gerais": "MG", "parana": "PR", "santa catarina": "SC",
               "rio grande do sul": "RS", "bahia": "BA", "goias": "GO", "espirito santo": "ES", "distrito federal": "DF",
               "pernambuco": "PE", "ceara": "CE"}
HEAVY_WORDS = ["bulky", "furniture", "appliance", "home goods", "sofa", "mattress", "fridge", "oversize"]
HEAVY_PHRASE = re.compile(r"heavy (goods|items|freight|products|parcels|cargo|packages)")  # NOT "heavy rains"
UF_NAME = {v: k.title() for k, v in STATE_NAMES.items()}
CAUSES = {"rain": "heavy rains", "flood": "flooding", "strike": "a transport strike", "storm": "severe storms",
          "landslide": "landslides", "fire": "a warehouse incident", "closure": "road closures", "highway": "highway closures"}


def norm(s): return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


@dataclass
class Disruption:
    raw_query: str
    origin_states: list = field(default_factory=list)
    origin_cities: list = field(default_factory=list)
    duration_days: int = 3
    scope: str = "all"              # all | heavy
    product_keywords: list = field(default_factory=list)
    cause: str = "a logistics disruption"
    parsed_by: str = "rules"
    notes: list = field(default_factory=list)

    def region_label(self):
        return ", ".join(c.title() for c in self.origin_cities) or " and ".join(UF_NAME.get(u, u) for u in self.origin_states) or "unknown"

    def to_dict(self): return asdict(self)


def rule_parse(q: str, known_cities: dict) -> Disruption:
    n = norm(q); d = Disruption(raw_query=q)
    m = re.search(r"(\d+)\s*(day|dia|week|semana)", n)
    if m: d.duration_days = int(m.group(1)) * (7 if m.group(2) in ("week", "semana") else 1)
    elif "week" in n: d.duration_days = 7
    else: d.notes.append("duration not found -> assumed 3 days")
    explicit_city = bool(re.search(r"\bcity\b|\bcidade\b", n))
    for name, uf in STATE_NAMES.items():
        if re.search(rf"\b{name}\b", n):
            d.origin_states.append(uf)
    for city, uf in known_cities.items():                      # city-level detection from real seller cities
        if city in STATE_NAMES and not explicit_city: continue  # ambiguity rule: 'Sao Paulo' => state
        if re.search(rf"\b{re.escape(city)}\b", n):
            d.origin_cities.append(city)
            if uf not in d.origin_states: d.origin_states.append(uf)
    if d.origin_cities: d.origin_states = []                    # city mention => narrower city filter
    for tok in re.findall(r"\b([A-Z]{2})\b", q):
        if tok in UF and tok not in d.origin_states and not d.origin_cities: d.origin_states.append(tok)
    kw = [w for w in HEAVY_WORDS if w in n]
    if (kw or HEAVY_PHRASE.search(n)) and not re.search(r"\ball (freight|shipments|orders|parcels)\b", n):
        d.scope = "heavy"; d.product_keywords = kw
    for k, v in CAUSES.items():
        if k in n: d.cause = v; break
    if "sao paulo" in n and not explicit_city:
        d.notes.append("'Sao Paulo' interpreted as the state (SP) - say 'city of Sao Paulo' to narrow")
    return d


def parse(q: str, llm, known_cities: dict) -> Disruption:
    d = rule_parse(q, known_cities)
    if llm and llm.available and os.getenv("LLM_PARSER") == "1":
        out = llm.complete(
            "You extract structured data from supply-chain disruption alerts. Reply with ONLY a JSON object: "
            '{"origin_states": [Brazilian UF codes], "origin_cities": [lowercase, no accents], "duration_days": int, '
            '"scope": "all" or "heavy", "product_keywords": [str], "cause": short phrase}. '
            'Use scope "heavy" only if the alert limits the impact to bulky/heavy goods. A bare mention of Sao Paulo means the state SP.',
            q, max_tokens=300, temperature=0)
        try:
            j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            sts = [s for s in j.get("origin_states", []) if s in UF]
            dur = int(j.get("duration_days", d.duration_days))
            if (sts or j.get("origin_cities")) and 1 <= dur <= 60:
                d.parsed_by = "llm+rules"
                # Rules are authoritative where they found something; the LLM only fills gaps.
                if not (d.origin_states or d.origin_cities):
                    d.origin_states, d.origin_cities = sts, [norm(c) for c in j.get("origin_cities", [])]
                if not re.search(r"\d+\s*(day|dia|week|semana)", norm(q)):
                    d.duration_days = dur
                if d.cause == "a logistics disruption":
                    d.cause = j.get("cause") or d.cause        
        except Exception:
            d.notes.append("LLM parse invalid -> used rule-based parse")
    return d

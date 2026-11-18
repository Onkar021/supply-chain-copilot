"""The Orchestrator: a plan -> execute -> verify pipeline with a full audit trace.

  Parser -> Retriever(RAG) -> Logistics Analyst -> Mitigation -> Aggregator/Narrator -> Customer Success

Each step reads only the previous step's typed output, and logs what it did (explainability).
One adaptive 're-plan' exists: if a city-level search finds nothing, widen to the whole state."""
import os
import json, time
from dataclasses import dataclass, field
import pandas as pd
from . import config as C
from .data_loader import build_corpus
from .retriever import OrderIndex, apply_scope
from .llm import LLM
from . import agents_parser, agents_logistics, agents_mitigation, agents_customer


@dataclass
class Report:
    query: str
    disruption: dict
    stats: dict
    impacted: pd.DataFrame
    sellers: pd.DataFrame
    categories: pd.DataFrame
    narrative: str
    emails: list
    trace: list
    llm_name: str
    retrieval_backend: str

    def to_markdown(self) -> str:
        s, d = self.stats, self.disruption
        md = [f"# Disruption Impact Report\n", f"**Alert:** {self.query}\n",
              f"**Interpreted as:** origin = {', '.join(d['origin_cities'] or d['origin_states'])}, {d['duration_days']} days, "
              f"scope = {d['scope']}, cause = {d['cause']} _(parsed by {d['parsed_by']})_  \n"
              f"**Snapshot date:** {C.AS_OF:%d %b %Y} | **LLM:** {self.llm_name} | **Retrieval:** {self.retrieval_backend}\n",
              "## Executive summary\n", self.narrative + "\n", "## Key numbers\n",
              f"| Metric | Value |\n|---|---|\n| Active orders retrieved from origin | {s['retrieved']:,} |\n"
              f"| Actually delayed | {s['delayed']:,} |\n| Absorbed by SLA buffer (no action) | {s['absorbed']:,} |\n"
              f"| **Will miss promised date** | **{s['breaching']:,}** |\n| Heavy/bulky among breaching | {s['breaching_heavy']:,} |\n"
              f"| Order value at risk | R$ {s['value_at_risk']:,.0f} |\n| Goodwill voucher budget | R$ {s['voucher_budget']:,.0f} |\n"
              f"| Extra reroute freight | R$ {s['reroute_cost']:,.0f} |\n",
              "**Severity tiers:** " + ", ".join(f"{k}: {v}" for k, v in s["tiers"].items()),
              "\n\n**Actions:** " + ", ".join(f"{k}: {v}" for k, v in s["actions"].items()) + "\n",
              "## Most impacted sellers\n", self.sellers.head(8).to_markdown(index=False) if not self.sellers.empty else "_none_",
              "\n## Most impacted categories\n", self.categories.head(8).to_markdown(index=False) if not self.categories.empty else "_none_",
              "\n## Highest-severity orders\n"]
        if not self.impacted.empty:
            top = self.impacted[self.impacted.action != "NO_ACTION"].sort_values("severity", ascending=False).head(8)
            md.append(top[["order_ref", "seller_state", "customer_state", "total_weight_kg", "delay_days", "breach_days", "residual_breach", "severity",
                           "tier", "action", "voucher_brl"]].round(1).to_markdown(index=False))
        md.append("\n## Sample customer emails\n")
        for e in self.emails:
            md += [f"### Order #{e['order_ref']} - action `{e['action']}` (source: {e['source']})\n", f"**Subject:** {e['subject']}\n",
                   "```\n" + e["body"] + "\n```\n"]
        md.append("## Agent trace (audit log)\n")
        md += [f"{i+1}. **{t['agent']}** - {t['detail']} _({t['ms']} ms)_" for i, t in enumerate(self.trace)]
        return "\n".join(md)


class Pipeline:
    def __init__(self):
        self.df, self.buffers = build_corpus()
        self.index = OrderIndex(self.df)
        self.llm = LLM()
        self.cities = (self.df.groupby("seller_city").seller_state.agg(lambda s: s.mode().iat[0])).to_dict()

    def run(self, query: str, lang="en", n_emails=2) -> Report:
        trace = []
        def step(agent, fn, fmt):
            t = time.time(); out = fn(); trace.append({"agent": agent, "detail": fmt(out), "ms": int((time.time() - t) * 1000)}); return out

        # 1 PARSE
        d = step("Disruption Parser", lambda: agents_parser.parse(query, self.llm, self.cities),
                 lambda d: f"states={d.origin_states} cities={d.origin_cities} days={d.duration_days} scope={d.scope} by={d.parsed_by} {d.notes}")

        # 2 RETRIEVE (hybrid RAG) + adaptive re-plan
        def retrieve(d):
            q = (f"{'bulky heavy furniture home goods appliances ' if d.scope == 'heavy' else ''}"
                 f"{' '.join(d.product_keywords)} freight shipments from {d.region_label()} awaiting dispatch")
            hits = self.index.hybrid_search(q, d.origin_states, d.origin_cities)
            if hits.empty and d.origin_cities:
                st = sorted({self.cities[c] for c in d.origin_cities if c in self.cities})
                d.notes.append(f"re-plan: no orders for city -> widened to state {st}")
                d.origin_states, d.origin_cities = st, []
                hits = self.index.hybrid_search(q, d.origin_states, None)
            return apply_scope(hits, d.scope, d.product_keywords)
        hits = step("Retriever (hybrid RAG)", lambda: retrieve(d),
                    lambda h: f"{len(h)} active orders in scope; backend={self.index.backend}; filter->dense+BM25->RRF")

        # 3 ANALYSE
        an = step("Logistics Analyst", lambda: agents_logistics.analyse(hits, d, self.buffers) if len(hits) else hits,
                  lambda a: f"{int((a.delay_days > 0).sum()) if len(a) else 0} delayed, {int((a.breach_days > 0).sum()) if len(a) else 0} breach SLA")
        an = an[an.delay_days > 0] if len(an) else an

        # 4 MITIGATE
        mit = step("Mitigation Agent", lambda: agents_mitigation.mitigate(an),
                   lambda m: f"actions={m.action.value_counts().to_dict() if len(m) else {}}")

        # 5 AGGREGATE + NARRATE
        stats, sellers, cats = self._aggregate(hits, mit)
        narrative = step("Impact Summarizer", lambda: self._narrate(d, stats, sellers),
                         lambda n: f"{self.narr_src} narrative, {len(n.split())} words")

        # 6 CUSTOMER EMAILS
        def emails():
            if mit.empty: return []
            return [agents_customer.draft(r, d, self.llm, lang) for r in agents_customer.pick_samples(mit, n_emails).itertuples()]
        em = step("Customer Success Agent", emails,
                  lambda e: f"{len(e)} drafts; sources={[x['source'] for x in e]}; guardrail issues={[x['guardrail_issues'] for x in e]}")
        return Report(query, d.to_dict(), stats, mit, sellers, cats, narrative, em, trace, self.llm.name, self.index.backend)

    # ------------------------------------------------------------------
    def _aggregate(self, hits, mit):
        n_ret = len(hits); delayed = len(mit)
        br = mit[mit.breach_days > 0] if delayed else mit
        stats = {"retrieved": n_ret, "delayed": delayed, "absorbed": int((mit.breach_days == 0).sum()) if delayed else 0,
                 "breaching": len(br), "breaching_heavy": int(br.heavy.sum()) if len(br) else 0,
                 "value_at_risk": float(br.items_value.sum()) if len(br) else 0.0,
                 "voucher_budget": float(mit.voucher_brl.sum()) if delayed else 0.0,
                 "reroute_cost": float(mit.extra_cost_brl.sum()) if delayed else 0.0,
                 "tiers": mit.tier.value_counts().to_dict() if delayed else {}, "actions": mit.action.value_counts().to_dict() if delayed else {}}
        if not len(br): return stats, pd.DataFrame(), pd.DataFrame()
        sellers = br.groupby(["seller_id"]).agg(city=("seller_city", "first"), state=("seller_state", "first"), orders=("order_id", "count"),
                                                heavy=("heavy", "sum"), avg_severity=("severity", "mean"), value_brl=("items_value", "sum")) \
                    .sort_values(["orders", "avg_severity"], ascending=False).round(1).reset_index()
        sellers["seller_id"] = sellers.seller_id.str[:8]
        cats = br.explode("categories").groupby("categories").agg(orders=("order_id", "nunique"), avg_weight_kg=("total_weight_kg", "mean"),
                                                                    avg_severity=("severity", "mean")).sort_values("orders", ascending=False).round(1).reset_index()
        return stats, sellers, cats

    def _narrate(self, d, s, sellers):
        self.narr_src = "template"
        top = sellers.head(3).to_dict("records") if len(sellers) else []
        if self.llm.available and os.getenv("LLM_NARRATIVE") == "1":
            out = self.llm.complete("You are a supply-chain control-tower analyst. Write a 5-7 sentence executive impact summary for an ops "
                                    "manager. Start directly with the summary, no preamble. All money is in Brazilian reais (R$). "
                                    "State how many orders are in scope and how many will miss their promised date. Use ONLY the numbers "
                                    "provided; do not invent figures. End with the 2 most important recommended actions.",
                                    json.dumps({"disruption": d.to_dict(), "stats": s, "top_sellers": top}, default=str), max_tokens=450)
            if out:
                out = out.strip()
                if out.lower().startswith("here is"):
                    out = out.split("\n", 1)[-1].strip()
                if str(s["breaching"]) in out and f"{s['retrieved']:,}" in out and "$" not in out.replace("R$", ""):
                    self.narr_src = "llm"
                    return out          # passed our checks; otherwise fall through to the template        
        pct = (100 * s["breaching"] / s["retrieved"]) if s["retrieved"] else 0
        txt = (f"{d.cause.capitalize()} in {d.region_label()} for {d.duration_days} days puts **{s['retrieved']:,}** active orders in scope. "
               f"**{s['delayed']:,}** will be delayed, but the SLA buffer absorbs {s['absorbed']:,} of them; **{s['breaching']:,}** ({pct:.0f}%) will miss their promised date, "
               f"{s['breaching_heavy']:,} of which are bulky/heavy goods, putting R$ {s['value_at_risk']:,.0f} of order value at risk. ")
        if top:
            txt += "Most exposed sellers: " + ", ".join(f"{t['seller_id']} ({t['city'].title()}, {t['orders']} orders)" for t in top) + ". "
        txt += (f"Recommended: (1) proactively notify the {s['breaching']:,} affected customers today with the new dates; "
                + ("(2) hold heavy freight at origin and reroute only light parcels" if s["breaching_heavy"]
                   else "(2) reroute the light parcels via an alternate hub")
                + f" (extra freight ~R$ {s['reroute_cost']:,.0f}); "
                f"budget ~R$ {s['voucher_budget']:,.0f} in goodwill vouchers.")
        return txt

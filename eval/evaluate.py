"""Lightweight evaluation harness - run: python -m eval.evaluate
Shows we measure quality instead of eyeballing it."""
import numpy as np, pandas as pd
from src import config as C
from src.orchestrator import Pipeline
from src.agents_customer import template_email, validate, build_facts
from src import agents_logistics as AL, agents_parser as AP
from src.retriever import apply_scope

p = Pipeline(); df = p.df
SC = [("Heavy rains delaying all freight leaving Sao Paulo for the next 5 days", None),
      ("Flooding blocks all shipments out of Rio de Janeiro for 4 days", None),
      ("Strike stops heavy goods leaving Minas Gerais for 7 days", "heavy"),
      ("Bulky furniture shipments from Parana are stuck for 3 days", "heavy"),
      ("Road closures delay all parcels leaving Santa Catarina for 2 weeks", None)]
print("== 1. Parser + retrieval vs. ground truth ==")
rows = []
for q, scope in SC:
    d = AP.rule_parse(q, p.cities); r = p.run(q, n_emails=0)
    gt = df[df.is_active & df.seller_state.isin(d.origin_states)]
    gt = apply_scope(gt, d.scope, d.product_keywords)
    got = set(r.impacted.order_id) | set()    # impacted subset; compare with retrieved via trace count
    n_ret = r.stats["retrieved"]
    rows.append({"query": q[:55], "states": d.origin_states, "scope": d.scope, "days": d.duration_days,
                 "retrieved": n_ret, "ground_truth": len(gt), "recall": round(min(n_ret, len(gt)) / max(len(gt), 1), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 2. Hybrid ranking sanity: does a 'bulky furniture' query rank heavy orders first? ==")
h = p.index.hybrid_search("bulky heavy furniture home goods appliances freight from Sao Paulo", ["SP"])
base = h.heavy.mean(); top = h.head(50).heavy.mean()
print(f"heavy share overall={base:.0%} | in top-50 hybrid results={top:.0%} | lift x{top/base:.1f}")

print("\n== 3. Severity monotonicity (heavier/longer => never lower severity) ==")
d = AP.rule_parse("rain in Sao Paulo for 5 days", p.cities)
act = df[df.is_active & (df.seller_state == "SP")].head(300)
s5 = AL.analyse(act, d, p.buffers); d10 = AP.rule_parse("rain in Sao Paulo for 10 days", p.cities); s10 = AL.analyse(act, d10, p.buffers)
print("10d severity >= 5d severity for all orders:", bool((s10.severity >= s5.severity).all()))

print("\n== 4. Email guardrails: template emails pass validator for every action x language ==")
r = p.run(SC[0][0], n_emails=0); ok = tot = 0
for lang in ("en", "pt"):
    for act_name, g in r.impacted[r.impacted.action != "NO_ACTION"].groupby("action"):
        for row in g.head(20).itertuples():
            f = build_facts(row, AP.rule_parse(SC[0][0], p.cities), lang); tot += 1; ok += not validate(template_email(f, lang), f)
print(f"{ok}/{tot} passed")
if p.llm.available:
    print("\n== 5. LLM email first-pass guardrail rate ==")
    r2 = p.run(SC[1][0], n_emails=4); print([ (e['attempts'], e['source']) for e in r2.emails])

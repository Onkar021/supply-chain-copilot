"""Streamlit demo UI:  streamlit run src/app.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import streamlit as st
from src.orchestrator import Pipeline
from src.cli import SCENARIOS
from src import config as C

st.set_page_config(page_title="Supply Chain Disruption Copilot", layout="wide")


@st.cache_resource(show_spinner="Loading orders, building the vector index (first run only)...")
def load(): return Pipeline()


p = load()
st.title("Supply Chain Disruption Copilot")
st.caption("Multi-agent decision system on Olist orders: Parser -> Hybrid RAG -> Logistics Analyst -> Mitigation -> Customer Success")

with st.sidebar:
    st.subheader("Settings")
    lang = st.radio("Email language", ["en", "pt"], horizontal=True)
    n_emails = st.slider("Sample emails", 1, 4, 2)
    st.markdown(f"**LLM:** {p.llm.name}  \n**Retrieval:** {p.index.backend}  \n**Orders indexed:** {len(p.df):,}  \n**Snapshot:** {C.AS_OF:%d %b %Y}")
    st.subheader("Example scenarios")
    for i, s in enumerate(SCENARIOS):
        if st.button(f"Scenario {i+1}", use_container_width=True): st.session_state.q = s

q = st.text_area("Describe the disruption", st.session_state.get("q", SCENARIOS[0]), height=80)
if st.button("Analyse impact", type="primary"):
    with st.spinner("Agents working..."):
        r = p.run(q, lang, n_emails)
    s = r.stats
    c = st.columns(5)
    c[0].metric("Orders in scope", f"{s['retrieved']:,}"); c[1].metric("Delayed", f"{s['delayed']:,}")
    c[2].metric("Absorbed by SLA", f"{s['absorbed']:,}"); c[3].metric("Will miss SLA", f"{s['breaching']:,}")
    c[4].metric("Value at risk", f"R$ {s['value_at_risk']:,.0f}")
    st.subheader("Executive summary"); st.write(r.narrative)
    t1, t2, t3, t4, t5 = st.tabs(["Impact", "Orders", "Customer emails", "Reasoning trace", "Interpretation"])
    with t1:
        a, b = st.columns(2)
        a.markdown("**Severity tiers**"); a.bar_chart(s["tiers"])
        b.markdown("**Mitigation actions**"); b.bar_chart(s["actions"])
        a.markdown("**Top sellers**"); a.dataframe(r.sellers.head(10), hide_index=True)
        b.markdown("**Top categories**"); b.dataframe(r.categories.head(10), hide_index=True)
    with t2:
        if not r.impacted.empty:
            cols = ["order_ref", "seller_city", "seller_state", "customer_state", "categories", "total_weight_kg", "status_asof", "delay_days",
                    "breach_days", "severity", "tier", "action", "new_eta", "voucher_brl", "rationale", "severity_why"]
            st.dataframe(r.impacted[cols].sort_values("severity", ascending=False), hide_index=True, use_container_width=True)
    with t3:
        for e in r.emails:
            with st.expander(f"Order #{e['order_ref']} - {e['action']} ({e['source']})", expanded=True):
                st.markdown(f"**Subject:** {e['subject']}"); st.text(e["body"])
                st.caption(f"Grounding facts given to the writer: {e['facts']}")
    with t4:
        for i, t in enumerate(r.trace, 1): st.markdown(f"**{i}. {t['agent']}** ({t['ms']} ms)  \n{t['detail']}")
    with t5: st.json(r.disruption)

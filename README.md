# Supply Chain Disruption Copilot

An AI-assisted **decision system (not a chatbot)** built on the Olist Brazilian E-Commerce dataset.
Give it a disruption alert, for example *"Heavy rains are delaying all freight leaving Sao Paulo for the next 5 days"*, and it:

1. finds exactly which **active orders, sellers and products** are affected (hybrid RAG),
2. quantifies the **delay, SLA breach and severity** of each order (deterministic analyst),
3. decides an **action per order**: no action, reroute, delay-and-notify, or offer refund (policy engine),
4. drafts **personalised, validated customer emails** (LLM with guardrails), and
5. returns an **impact summary with a full audit trail** (CLI and Streamlit UI).

**Design principle:** *deterministic core, LLM at the edges.* Numbers, money and promises come from auditable code.
The LLM only writes language, and everything it writes is validated before use.

---

## 1. Results on real Olist data (snapshot 15 Jun 2018)

| | Scenario 1: rain, Sao Paulo, 5 days, all freight | Scenario 2: strike, MG + PR, 10 days, bulky furniture/appliances |
|---|---|---|
| Active orders in scope | 1,543 | 69 |
| Delayed | 581 | 34 |
| Absorbed by SLA buffer (no action) | 525 | 18 |
| **Will miss promised date** | **56 (4%)** | **16 (23%)** |
| Heavy/bulky among breaching | 7 | 9 |
| Order value at risk | R$ 15,685 | R$ 4,377 |
| Goodwill voucher budget | R$ 569 | R$ 222 |
| Extra reroute freight | R$ 349 | R$ 56 |
| Actions | 525 no-action, 45 reroute, 7 delay, 4 refund-offer | 18 no-action, 10 delay, 5 reroute, 1 refund-offer |

Full outputs, including sample emails: [`samples/scenario_1.md`](samples/scenario_1.md), [`samples/scenario_2.md`](samples/scenario_2.md).
Screenshots of the UI: `docs/screenshots/`.

**Key insight:** Olist promised delivery dates are heavily padded (median slack about 7 days at the 25th percentile).
A 5-day disruption delays 581 orders, but only 56 actually breach their promise. The system avoids notifying the other 525 customers.

---

## 2. Architecture

```
 "Heavy rains delaying all freight leaving Sao Paulo for 5 days"
            |
 [0] DISRUPTION PARSER      rules (authoritative); optional LLM fills gaps only
            |                -> origin states/cities, duration, scope (all|heavy), cause
 [1] RETRIEVER (hybrid RAG) metadata filter (geography + active)  -> candidate set (full recall)
            |                dense (ChromaDB + MiniLM) + BM25 -> Reciprocal Rank Fusion
            |                scope filter (bulky weight/volume OR categories the alert names)
 [2] LOGISTICS ANALYST      deterministic: delay days, SLA breach (disruption-caused only), severity 0-100
            |
 [3] MITIGATION AGENT       deterministic policy: NO_ACTION / REROUTE / DELAY / CANCEL_OFFER + voucher + cost
            |
 [4] SUMMARIZER             aggregates (sellers, categories, tiers, money); template narrative
            |                (optional LLM rewrite, accepted only if its numbers verify)
 [5] CUSTOMER SUCCESS       LLM drafts from decided facts -> validators -> retry once -> template fallback
            |
   Report: impact summary + mitigation table + emails + agent trace
   Interfaces: CLI (src/cli.py) and Streamlit UI (src/app.py)
```

| File | Role |
|---|---|
| `src/config.py` | **All assumptions and thresholds in one place** |
| `src/data_loader.py` | Joins Olist tables to one row per order, snapshot logic, learns SLA slack from history |
| `src/retriever.py` | ChromaDB + sentence-transformers, BM25, RRF fusion, TF-IDF fallback, scope filter |
| `src/agents_parser.py` | Disruption parser |
| `src/agents_logistics.py` | Delay, SLA breach, severity |
| `src/agents_mitigation.py` | Policy engine |
| `src/agents_customer.py` | Email drafting, validators, templates (EN/PT) |
| `src/orchestrator.py` | Pipeline, audit trace, aggregation, report |
| `src/llm.py` | Swappable LLM wrapper: Anthropic, Ollama, or none |
| `src/synthetic.py` | Olist-shaped synthetic data so the project runs with no download |
| `eval/evaluate.py` | Evaluation harness |

---

## 3. Quick start

### Prerequisites
Python 3.10 to 3.12. About 3 GB disk and 8 GB RAM. On Windows, install the
[Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) if PyTorch fails to load (`shm.dll` error).

### Install
```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Light install (no embeddings, TF-IDF retrieval): `pip install pandas numpy scikit-learn tabulate streamlit requests`

### Get the data (free Kaggle account)
Download **Brazilian E-Commerce Public Dataset by Olist**: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
and copy these files into `data/raw/`:
`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_products_dataset.csv`,
`olist_sellers_dataset.csv`, `olist_customers_dataset.csv`, and optionally `product_category_name_translation.csv`.
If they are missing, the app generates **synthetic Olist-shaped data** so everything still runs (demo mode).
The dataset is licensed CC BY-NC-SA 4.0, so it is not included in this repository.
After changing data, delete `data/processed/` and `data/chroma/`.

### Run
```bash
# CLI
python -m src.cli "Heavy rains are delaying all freight leaving Sao Paulo for the next 5 days"
python -m src.cli "<alert>" --lang pt --emails 3

# Web UI  ->  http://localhost:8501
streamlit run src/app.py

# Regenerate the two sample reports in samples/
python -m src.cli --samples --emails 1

# Evaluation
python -m eval.evaluate
```
First run: builds the order corpus (about 30 s) and embeds 9,816 orders once with MiniLM (1 to 3 min on CPU, about 90 MB model download). Later runs start in seconds.
The report header shows `Retrieval: chroma` (embeddings) or `tfidf` (fallback).

### Optional: free local LLM (no subscription)
```bash
# install Ollama from https://ollama.com/download, then:
ollama pull llama3.2:3b            # about 2 GB; use llama3.1:8b for better emails if you have 16 GB RAM
```
```powershell
$env:USE_OLLAMA="1"; $env:OLLAMA_MODEL="llama3.2:3b"        # Windows PowerShell
```
```bash
export USE_OLLAMA=1 OLLAMA_MODEL=llama3.2:3b                # macOS / Linux
```
Or use the Anthropic API: `ANTHROPIC_API_KEY=...`. Without any LLM, emails come from validated templates.
On CPU, a 3B model takes roughly 1 to 2 minutes per email, so use `--emails 1` or 2.

| Variable | Purpose |
|---|---|
| `USE_OLLAMA=1`, `OLLAMA_MODEL`, `OLLAMA_URL` | Use a local Ollama model |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Use the Anthropic API |
| `LLM_PARSER=1` | Let the LLM fill gaps in alert parsing (off by default, see section 6) |
| `LLM_NARRATIVE=1` | Let the LLM rewrite the executive summary (off by default, verified before use) |
| `FORCE_TFIDF=1` | Skip embeddings (fast, no PyTorch) |
| `AS_OF_DATE`, `MAX_ORDERS` | Snapshot date (default 2018-06-15) and corpus cap (default 10,000) |

Note: `.env` is only read by Docker Compose. For local runs, set the variables in your shell.

### Docker (experimental)
```bash
cp .env.example .env
docker compose up --build          # UI on http://localhost:8501
```
The container cannot reach a host-side Ollama by default (set `OLLAMA_URL=http://host.docker.internal:11434`).
For the LLM demo, run locally.

---

## 4. How it reasons (multi-step chain)

**Geography -> sellers -> orders -> SKU weight -> delay -> SLA breach -> action -> words.**

1. **Parse.** "Sao Paulo / 5 days / all freight" becomes `{states:[SP], days:5, scope:all, cause:heavy rains}`.
   A bare "Sao Paulo" means the state; "city of X" narrows to a city. A city search that finds nothing is automatically widened to the state (re-plan, logged in the trace).
2. **Retrieve.** Each order is a natural-language document. A hard metadata filter (active, seller in origin) defines the impacted set with full recall.
   Dense similarity (ChromaDB cosine, MiniLM-L6-v2, 384-d) and BM25 (k1=1.5, b=0.75) each rank the candidates, and Reciprocal Rank Fusion (k=60) merges them.
   For heavy-goods alerts, the scope filter keeps orders that are bulky by weight/volume, or whose categories the alert names (for example "furniture" maps to `furniture_*`).
3. **Analyse (deterministic).**
   - *Delay:* an order still at the seller is held from its expected dispatch (`shipping_limit_date`, clipped to now) until the window ends, plus a backlog-recovery allowance of 25% of the window.
     Parcels picked up within the last 2 days lose 50% of the window. Everything else has already left the origin network.
   - *SLA:* baseline ETA = promised date minus the slack orders historically have (25th percentile by destination state, learned from delivered orders; global value 7.0 days).
     Projected ETA = baseline + delay. **Breach counts only the delay the disruption adds** (never more than the delay days). Orders already late before the snapshot are flagged `already_late`.
   - *Severity 0 to 100:* SLA breach (50) + weight (25) + order value (15) + freight burden (10). Tiers: Critical >= 70, High >= 50, Medium >= 30, otherwise Low; "Absorbed" if no breach.
4. **Decide (policy engine).**
   - No breach: **NO_ACTION** (monitor only).
   - Light parcel (<= 5 kg), still at seller, delay >= 3 days: **REROUTE** via alternate hub (+35% freight, +2 days vs normal). Residual breach is measured after rerouting.
   - Heavy/bulky or already in transit: **DELAY** and notify. Rerouting heavy freight costs more than it saves, and parcels in a carrier network cannot be pulled back.
   - Residual breach >= 7 days and not yet shipped: **CANCEL_OFFER** (customer chooses refund or waiting, never auto-cancelled).
   - Goodwill voucher: 5/10/15% of item value for residual breach <= 3 / <= 7 / > 7 days, capped at R$ 50.
5. **Write.** The email agent receives only facts already decided upstream (order, dates, action, voucher, refund eligibility, already-late flag).

---

## 5. Key design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **LLM only for language; rules for numbers, money, policy** | Refunds and dates must be auditable, reproducible and testable | Policy changes need a config/code change (which is good governance) |
| **Hard filter first, then semantic ranking** | Impact analysis needs *all* affected orders; top-k vector search silently misses some | Embeddings rank and prioritise rather than define membership |
| **Hybrid dense + BM25 + RRF** | Dense catches meaning ("sofa" ~ furniture), BM25 catches exact tokens (cities, IDs); RRF needs no score calibration | More code than dense-only |
| **Rules authoritative in the parser** | A 3B model misread "heavy rains" as heavy goods and shrank the impact 10x (see section 7) | Less flexible on novel phrasings; LLM parsing is opt-in |
| **SLA slack learned from history (conservative percentile)** | Promised dates are padded; ignoring that over-notifies customers | The percentile is a tunable judgement call |
| **Email guardrails + retry + template fallback** | Never send a wrong date, voucher or invented offer; works with no LLM | Template emails are less personal |
| **Template summary by default** | Always numerically correct and instant | Less fluent than an LLM summary |
| **Plain pipeline, not a free-form agent loop** | Predictable, debuggable, cheap, easy to explain | Less flexible than an LLM planner (see roadmap) |
| **Proposes, does not execute** | Human-in-the-loop for refunds and rerouting | Needs an approval queue in production |
| **Graceful degradation everywhere** | No LLM, no embeddings, no dataset: it still runs | More code paths to test |

LLM guardrails (`src/agents_customer.py`): order reference and new date present; exact voucher amount present, and no voucher or refund mentioned when not approved;
refund option present when approved; no unfilled placeholders; apology present; no banned phrases ("guarantee", "100%");
already-late orders must acknowledge it, and others must not claim it; the voucher is not tied to replying; length 50 to 230 words.
On failure the LLM gets the error list and retries once, then a template that always passes the validator is used.

---

## 6. Assumptions (the dataset has no live logistics data)

| # | Assumption | Effect |
|---|---|---|
| A1 | **Replay clock:** snapshot `AS_OF` = 2018-06-15. "Active" = purchased in the prior 45 days, not delivered or cancelled on that date. Corpus capped at 10,000 (9,816 orders, 2,241 active) | Makes historical data behave like a live order book |
| A2 | `awaiting_dispatch` if carrier pickup is after the snapshot, otherwise `in_transit` (uses historical timestamps as ground truth for the replay) | Drives who can be rerouted or cancelled |
| A3 | "Freight leaving X" = orders whose **seller** is in X. A bare "Sao Paulo" = state SP | Geographic filter |
| A4 | Held orders wait until the window ends + 25% recovery; parcels picked up <= 2 days ago lose 50% of the window; the rest are unaffected | Delay model |
| A5 | `shipping_limit_date` approximates expected dispatch time | Per-order hold length |
| A6 | SLA slack = 25th percentile of (promised minus actual) delivery by destination state, learned from delivered orders (global 7.0 days) | Absorbs short delays |
| A7 | Heavy/bulky = total weight >= 8 kg or volume >= 60 L. Missing weight/dimensions imputed with the category median | Reroute policy |
| A8 | Reroute: +35% freight, +2 days, only <= 5 kg parcels with delay >= 3 days | Cost/benefit rule |
| A9 | Vouchers 5/10/15% of item value (cap R$ 50). Refund option only if residual breach >= 7 days and not shipped | Compensation policy |
| A10 | Breach and residual breach count only disruption-caused delay. Pre-existing lateness is flagged `already_late` and changes the email wording | Avoids blaming the disruption for existing delays |
| A11 | Olist customers are anonymised, so emails use a generic greeting (production: CRM merge field) | Limited personalisation |
| A12 | The Customers table is used (destination state) although the brief lists four tables | Needed for SLA slack |
| A13 | `samples/` were generated on the real Olist data (snapshot 15 Jun 2018) with `llama3.2:3b` in template-summary mode | Reproduce with `python -m src.cli --samples` |

All numeric assumptions live in `src/config.py`.

---

## 7. Lessons from validating on real data

Running on real Olist data (after building on synthetic data) exposed issues that a first pass missed:

1. **Pre-existing lateness was blamed on the disruption.** Orders already weeks late showed 22 "breach days" and got weather apologies.
   Fix: breach is capped at the delay the disruption adds, `already_late` is tracked, and emails acknowledge prior lateness.
2. **The heavy-goods scope was too loose.** "Bulky furniture" matched bed linen and housewares (scenario 2 went from 139 to 69 orders).
   Fix: keyword-to-category hints, with generic hints only when the alert names no product.
3. **Residual breach after rerouting included old lateness**, which pushed vouchers to the R$ 50 cap. Fix: measure against the later of the promised date and baseline ETA (voucher budget R$ 954 to R$ 569).
4. **A 3B LLM parser silently shrank the impact 10x** (read "heavy rains" as heavy goods and "Sao Paulo" as the city: 1,543 to 136 orders). Fix: rules are authoritative, and the LLM is opt-in and fills gaps only.
5. **LLM emails drifted in ways fact-checking can't see:** placeholders like `[Your Name]`, claiming the voucher needs a reply, invented "find it in your account details", wrongly calling an on-time order already late.
   Fix: stricter prompt and validators with retry and fallback. Remaining limit: a small model can still add unsupported phrasing.
6. **The reroute email promised "sooner" while the new date was later.** Fixed the wording to "limit the delay".

---

## 8. Evaluation (`python -m eval.evaluate`)

| Check | Result |
|---|---|
| Parser + retrieval vs ground truth (5 scenarios) | Coverage 1.0 in all (1543, 138, 74, 35, 87 orders) |
| Hybrid ranking: bulky-furniture query | Heavy orders are 10% overall but 96% of the top 50 (9.6x lift) |
| Severity monotonicity (10-day vs 5-day) | True for every order |
| Template emails pass the validator (every action x language) | 62/62 |
| LLM emails (llama3.2:3b, 2 emails) | 1 passed first try, 1 after a retry, 0 template fallbacks |

**Honest caveats:** the coverage check and the template check are consistency checks, not independent validation (the ground truth reuses the same filter code; templates are written to pass their own validator).
The ranking lift is partly by construction, since order documents contain the phrase "Bulky heavy freight". The LLM result is a tiny sample on a small local model.
Next step: hand-labelled orders for true recall, and an LLM-as-judge plus human rubric for email quality.

---

## 9. Limitations and roadmap

**Limitations:** no carrier, route or real-time data (delay and reroute parameters are assumptions, not estimates); orders are matched by seller state, not routes;
policy thresholds are not calibrated against actuals; a 3B model can still produce stilted emails; Docker is provided but experimental.

**Roadmap:** live weather and carrier APIs; route graph and capacity-aware rerouting; calibrate the delay model against historical disruptions;
cost-optimised voucher allocation; a separate service-recovery queue for orders already late; approval queue and an email service integration;
LLM tool-calling planner for novel disruptions (LangGraph); drift monitoring and A/B tests of email variants; pgvector/Qdrant and batch execution for millions of orders.

---

## 10. Project layout
```
src/            pipeline code (see table in section 2)
eval/           evaluation harness
samples/        generated reports for two scenarios
docs/           screenshots and presentation material
data/raw/       put Olist CSVs here (not committed)
Dockerfile, docker-compose.yml, requirements.txt, .env.example
```
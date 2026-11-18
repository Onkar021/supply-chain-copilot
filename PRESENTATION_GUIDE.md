# Presentation Guide: Supply Chain Disruption Copilot

Part A: technical walkthrough (know this cold). Part B: 10-minute rehearsal script. Part C: Q&A. Part D: backup plan and cheat sheet.

---

# Part A: Everything we built, in technical detail

## A1. The problem and the one-line answer
The brief: given a disruption alert, coordinate specialised agents to find impacted orders, quantify the damage, decide mitigation, and write customer communications. Not a chatbot.

**One-liner:** "A deterministic-core, LLM-at-the-edges pipeline. Hybrid RAG finds exactly which orders a disruption hits, analyst code quantifies SLA damage, a policy engine decides the action, and an LLM only writes the words, with validators around it."

## A2. Data layer (`data_loader.py`)
- **Tables:** orders, order_items, products, sellers, plus customers (for destination state) and the category translation file.
- **One row per order:** items are aggregated per order: item count, total weight (kg), total volume (cm3), item value, freight value, seller, latest `shipping_limit_date`, and the set of English category names.
- **Imputation:** missing product weight/volume filled with the category median, then the global median.
- **Replay clock.** Olist is historical, so I pick a snapshot, `AS_OF = 2018-06-15`, and treat it as "now".
  - Corpus = orders purchased in the 45 days before the snapshot, capped at 10,000 (result: 9,816 orders).
  - `is_active` = not cancelled/unavailable and not yet delivered at the snapshot (2,241 active).
  - `status_asof` = `awaiting_dispatch` if carrier pickup is after the snapshot, else `in_transit`, else `closed`.
  - This uses historical timestamps with hindsight. It is a deliberate simulation assumption.
- **Heavy flag:** total weight >= 8 kg or volume >= 60,000 cm3.
- **SLA slack learned from data:** for orders delivered before the snapshot, compute (promised date minus actual delivery) in days. Take the 25th percentile per destination state, clipped to 0..20 days. Global value on real data: 7.0 days.
  - Why the 25th percentile: conservative. Only count the slack that 75% of orders really have.
- Result is cached as a pickle, so restarts are fast.

## A3. Retrieval (`retriever.py`)
- **Documents:** each order becomes a sentence-style document: order ref, seller city/state, destination, categories, weight, "Bulky heavy freight" or "Light parcel", status, promised date.
- **Vector store:** ChromaDB `PersistentClient`, cosine HNSW, embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Metadata per order: `seller_state`, `seller_city`, `status`, `is_active`, `heavy`.
- **Why a hard filter first:** the question is "which orders are affected?" That needs complete recall. A top-k vector search would silently miss some. So:
  1. `collection.get(where={"$and":[{is_active:true},{seller_state:{"$in":[...]}}]})` returns the complete candidate set (the exact impacted set by definition).
  2. Dense ranking: `collection.query(..., n_results=len(candidates))` ranks every candidate by semantic similarity to the disruption text.
  3. Sparse ranking: my own BM25 (k1=1.5, b=0.75) over the candidates' text.
  4. **Reciprocal Rank Fusion:** score = 1/(60 + dense_rank) + 1/(60 + bm25_rank). It merges ranks, not raw scores, so there is no need to calibrate cosine against BM25.
- **Scope filter (`apply_scope`):** for "heavy" alerts, keep orders that are bulky by weight/volume, or whose categories match hints the alert names (a keyword map: "furniture" -> `furniture_*`, "appliance" -> `*appliance*`, etc.). Generic heavy-category hints apply only when the alert names no product type.
- **Fallback:** if PyTorch or Chroma fail to load, the same interface runs on TF-IDF (unigram+bigram) so the system still works.
- **Honest framing:** embeddings rank and prioritise; the hard filter defines membership. Say that proactively.

## A4. Agent 0: Disruption Parser (`agents_parser.py`)
- Rules extract: duration (regex: days/weeks, EN+PT), states (name -> UF map), cities (matched against the real seller-city list), scope (keywords like bulky/furniture/appliance or the phrase "heavy goods/freight"; not the word "heavy" alone, to avoid "heavy rains"), cause (rain/flood/strike/...).
- Ambiguity rule: bare "Sao Paulo" = state SP; "city of Sao Paulo" = city. Recorded in `notes`.
- **LLM is opt-in** (`LLM_PARSER=1`) and only fills what rules found nothing for. Reason: see lesson 4 in A9.

## A5. Agent 2: Logistics Analyst (`agents_logistics.py`, pure pandas, deterministic)
Let `dur` = disruption days, `end = AS_OF + dur`, `recov = ceil(0.25 * dur)`.
1. **Delay days**
   - Awaiting dispatch: `dispatch = max(shipping_limit_date, AS_OF)`. If `dispatch > end`, no delay. Otherwise `delay = days(end - dispatch) + recov`.
   - In transit and picked up within the last 2 days: `delay = ceil(0.5 * dur)` (still in the origin network).
   - Otherwise 0 (already left).
2. **SLA:** `baseline_eta = promised_date - slack(destination_state)`, clipped to at least `AS_OF + 1 day`. `projected_eta = baseline_eta + delay`.
   - `breach_days = max(0, ceil(projected_eta - promised_date))`, **then capped at `delay_days`**, so the disruption is never blamed for lateness it did not cause.
   - `already_late = promised_date < AS_OF` is flagged for email wording.
3. **Severity (0 to 100)** = breach (min(breach/10,1) x 50) + weight (min(kg/30,1) x 25) + value (min(R$/500,1) x 15) + freight burden (min((freight/value)/0.5,1) x 10).
   - Tiers: Not impacted (no delay), Absorbed (no breach; severity capped at 15), Critical >= 70, High >= 50, Medium >= 30, Low otherwise.
   - Each order stores `severity_why`, the component breakdown, for explainability.

## A6. Agent 3: Mitigation (`agents_mitigation.py`, rules)
In order:
1. `breach_days == 0` -> **NO_ACTION**.
2. Awaiting dispatch, not heavy, <= 5 kg, `delay >= 3` -> **REROUTE**. New ETA = `min(projected, baseline + 2 days)`. Extra cost = 35% of freight. **Residual breach** = days past `max(promised, baseline_eta)`, i.e. only what the disruption still causes after the reroute.
3. Otherwise hold. Residual breach >= 7 and awaiting dispatch -> **CANCEL_OFFER** (customer's choice, never auto-cancel). Else **DELAY**.
4. **Voucher** = item value x (5% if residual <= 3 days, 10% if <= 7, 15% beyond), capped at R$ 50.
5. Every row gets a human-readable `rationale`.
Why rules, not an LLM: money and promises must be deterministic, explainable and unit-testable.

## A7. Orchestrator and summary (`orchestrator.py`)
- A plain pipeline: parse, retrieve, analyse, mitigate, aggregate, summarise, draft emails. Each step is wrapped with timing and a one-line description, producing the **audit trace** shown in the UI.
- **One adaptive re-plan:** if a city-level search returns nothing, widen to the state and log it.
- **Aggregation:** totals, tiers, actions, value at risk (item value of breaching orders), voucher budget, reroute cost, top sellers and categories (computed on breaching orders).
- **Summary:** a deterministic template by default (always numerically right). `LLM_NARRATIVE=1` lets the LLM rewrite it, accepted only if it contains the correct retrieved and breaching counts and uses R$ rather than $.

## A8. Agent 4: Customer Success and the LLM layer (`agents_customer.py`, `llm.py`)
- **LLM wrapper:** Anthropic API if a key is set, else local Ollama if `USE_OLLAMA=1`, else none. Temperature 0.3. Every caller works when it returns `None`.
- **Prompt:** a system prompt (tone, rules, JSON output) plus the facts as JSON plus an explicit "MUST include" list generated from the facts: order reference, new date, exact voucher amount (or an explicit "no voucher"), refund option (or "no mention of refunds"), the voucher is already applied, acknowledge already-late orders, separate paragraphs.
- **Validators (deterministic):** reference and date present; voucher amount correct, and no voucher or refund mentioned if not approved; refund present when approved; no `[placeholders]`; apology present; banned phrases; no claim of prior lateness for on-time orders; acknowledgement of prior lateness when true; voucher not tied to replying; 50 to 230 words.
- **Control loop:** draft, validate, on failure re-prompt once with the error list, then fall back to a template. The templates are asserted to pass the validator.
- English and Brazilian Portuguese are both supported (templates and prompts).

## A9. Bugs the real data exposed (your best story)
1. **Pre-existing lateness blamed on the disruption.** An order 2 weeks late already showed 22 breach days and got a weather apology. Fix: cap breach at disruption-caused delay, add `already_late`, adapt wording.
2. **Scope too loose:** "bulky furniture" matched bed linen and housewares. Scenario 2 fell from 139 to 69 orders after keyword-to-category hints.
3. **Residual breach after reroute included old lateness** and hit the R$ 50 voucher cap. Fix: measure against `max(promised, baseline)`. Voucher budget R$ 954 to R$ 569.
4. **The 3B LLM parser read "heavy rains" as heavy goods and "Sao Paulo" as the city:** 1,543 orders became 136, and 56 breaches became 2. No error was raised. Fix: rules authoritative, LLM opt-in. This is the strongest "LLM output must be verified" example.
5. **LLM email drift:** `[Your Name]` placeholders, "reply to claim the voucher", invented "find it in your account details", wrongly calling an on-time order already late. Fix: stricter prompt and validators.
6. **Reroute email said "sooner" when the new date was later.** Wording fixed. Validators check facts, not every sentence. Say so.

## A10. Evaluation (`eval/evaluate.py`) and what it does and does not prove
- Five scenarios: parser + retrieval coverage = 1.0 (1543, 138, 74, 35, 87 orders). This checks consistency, not independent recall (the ground truth reuses the filter).
- Ranking lift: heavy orders are 10% of the pool and 96% of the top 50 (9.6x). Partly by construction (the document text says "Bulky heavy freight").
- Severity monotonicity: 10-day severity >= 5-day severity for every order.
- 62/62 template emails pass the validator (EN and PT, all actions). Templates are written to pass.
- LLM: 2 emails, one passed first try, one after one retry, zero fallbacks. Tiny sample, a small model.
- **Next:** hand-label about 100 orders for true recall; human rubric plus LLM-as-judge for email quality.

## A11. Results (real data)
Scenario 1 (rain, Sao Paulo, 5 days): 1,543 in scope, 581 delayed, 525 absorbed, **56 breach (4%)**, 7 heavy, R$ 15,685 at risk, R$ 569 vouchers, R$ 349 reroute freight; actions 525 / 45 reroute / 7 delay / 4 refund-offer.
Scenario 2 (strike, MG+PR, 10 days, bulky): 69 in scope, 34 delayed, 18 absorbed, **16 breach (23%)**, 9 heavy, R$ 4,377 at risk, R$ 222 vouchers.
Performance: after the one-off index build, retrieval is about 0.5 s and the whole deterministic pipeline is under 1 s. A 3B model on CPU takes about 1 to 2 minutes per email.

---

# Part B: 10-minute rehearsal script

Set up first: Streamlit open in template mode (or with Ollama and **Sample emails = 1**), README open in a second tab, the screenshots as backup. Practise the whole thing twice with a timer.

**0:00 to 1:00. Problem and answer**
> "Supply chain disruptions need decisions, not chat answers. I built a system that takes an alert like 'heavy rains delaying all freight leaving Sao Paulo for 5 days' and produces an impact summary, a per-order action and customer emails. The principle is deterministic core, LLM at the edges. Numbers and money come from auditable code, and the LLM only writes language, behind validators."

**1:00 to 3:00. Live demo (scenario 1)**
Click Analyse impact.
> "1,543 active orders ship from Sao Paulo sellers. 581 get delayed, but only 56 miss their promised date. Olist promises are padded, and I learned about 7 days of real slack from history. So 525 customers are not bothered."
Click **Orders**: sort by severity, open `rationale` and `severity_why`.
> "Every decision is explainable, with the component breakdown, and nothing here is an LLM guess."
Click **Customer emails**: point at the date, voucher, separate refund option, and the already-late acknowledgement.
Click **Reasoning trace** and **Interpretation**.
> "Each agent's output and timing is logged. Alert parsing is shown too."

**3:00 to 4:30. Architecture** (README diagram or whiteboard)
> "Six steps. Parser, retriever, logistics analyst, mitigation agent, summarizer, customer-success agent, and an orchestrator that logs a trace. The first four stages are deterministic. The LLM appears only for email drafting and, optionally, for parsing or summary rewriting. And everything it writes is validated."

**4:30 to 6:00. Retrieval**
> "Impact analysis needs complete recall, so geography and active status are a hard metadata filter in Chroma. Then I rank candidates by dense similarity, MiniLM embeddings, plus BM25 for exact tokens like city names, merged with Reciprocal Rank Fusion, so I never have to calibrate cosine against BM25. Embeddings prioritise and handle fuzzy product language, and the filter defines membership. If PyTorch is missing, it falls back to TF-IDF and still works."

**6:00 to 7:30. Reasoning chain and policy**
> "Geography to sellers to orders to weight to delay to SLA breach to action to words. An order still at the seller is held until the window closes plus 25% recovery; parcels already on the road are unaffected. Heavy freight is held, because rerouting it costs more than it saves. Light parcels still at the seller are rerouted. Long breaches get a refund option, never an auto-cancel. Vouchers are 5, 10 or 15% by tier, capped at R$ 50."

**7:30 to 9:00. What real data taught me** (your strongest section)
> "I built on synthetic data first, then validated on real Olist. Four things broke. One: orders already weeks late were blamed on the rain, so I cap breach at the delay the disruption adds. Two: 'bulky furniture' matched bed linen, so I added keyword-to-category scope. Three: after rerouting, old lateness inflated vouchers, so voucher budget fell from 954 to 569 reais. Four, and the best one: my local 3B model parsed 'heavy rains' as heavy goods and shrank the analysis from 1,543 orders to 136, silently. So rules are now authoritative and the LLM is opt-in. That is why every LLM output sits behind validators."

**9:00 to 10:00. Limits and next steps**
> "Be clear on limits: Olist has no carrier or route data, so my delay and reroute parameters are assumptions in one config file, not estimates. The evaluation shows consistency, not independent recall; next I'd hand-label orders and add an LLM-judge for email quality. In production: live weather and carrier APIs, route-aware rerouting, an approval queue before sending, and a LangGraph planner for novel disruptions."

If time is short, drop the architecture section. If a question goes long, say "happy to go deeper on that after".

---

# Part C: Likely questions and strong answers

**Why not let the LLM decide refunds and vouchers?** It is non-deterministic, can hallucinate, and gives no audit trail. Policy as code is testable and governable. The LLM only receives decided facts.

**Why not pure vector search?** Top-k misses affected orders. I need an exhaustive filter for membership, and embeddings for ranking and fuzzy product language.

**Why RRF?** It fuses ranks, so there is no need to normalise cosine against BM25 scores, and it works without tuning.

**Why is only 4% of the impact "will miss SLA"?** Promised dates are padded, and I learned conservative slack (25th percentile by destination state) from historical deliveries. A 5-day disruption plus recovery mostly stays inside the slack. The system avoids over-notifying.

**How do you know it works?** `eval/evaluate.py`: coverage, ranking lift, severity monotonicity, guardrail pass rates, and a tiny LLM check. I am explicit that these are consistency checks. Next: hand-labelled orders and a human/LLM-judge rubric.

**What if the LLM is down or weak?** Every LLM call has a fallback: templates for emails, rules for parsing, a template summary. I also tested with a weak 3B model and found real failures, which the validators caught.

**What did the LLM get wrong?** The parser shrank the scope 10x, and emails had placeholders, invented claims and wrong lateness statements. This is the reason for rules-first and validators.

**How would you scale to millions of orders?** Pre-filter in the warehouse (SQL), vector DB with metadata filters (Qdrant or pgvector), vectorised or Spark batch for the deterministic stages, LLM only for sampled or segmented emails with templates per segment.

**Why not LangGraph or CrewAI?** A fixed pipeline is simpler, cheaper and easier to debug. The typed step boundaries map directly onto LangGraph nodes if branching, loops or LLM planning become necessary.

**Which assumptions matter most?** The replay clock, the delay model (25% recovery, 2-day exposure), the slack percentile, the heavy threshold, and the policy values. All in `config.py`. I would calibrate them against carrier actuals and past disruptions.

**What about customer privacy?** Olist is anonymised and the emails use a generic greeting. In production, names and contact details come from a CRM, with PII kept out of prompts where possible.

**Why a local 3B model?** Zero cost and fully reproducible for a demo. It is slow (1 to 2 minutes per email on CPU) and stylistically stiff. In production I would use a stronger hosted model, with the same validators.

**Biggest weakness?** No real logistics data. The disruption model is assumption-driven. Everything else (filters, SLA math, policy) is transparent and tunable.

---

# Part D: Backup plan and cheat sheet

**If the live demo is slow or breaks:** run without the LLM (`Remove-Item Env:USE_OLLAMA`), which is instant. Or show the screenshots and `samples/*.md`. Say: "The LLM run takes about a minute per email on my CPU, so here are the saved results."

**Numbers to remember**
- 9,816 orders in corpus, 2,241 active, snapshot 15 Jun 2018, slack 7.0 days
- Scenario 1: **1,543 / 581 / 525 / 56**, R$ 15,685 at risk, R$ 569 vouchers
- Scenario 2: **69 / 34 / 18 / 16**, R$ 4,377 at risk, R$ 222 vouchers
- Retrieval lift 9.6x; template guardrails 62/62; LLM test 2 emails, 0 fallbacks
- Parser bug: 1,543 became 136 orders, and 56 breaches became 2

**Tech stack in one breath:** Python, pandas, ChromaDB, sentence-transformers MiniLM-L6-v2, custom BM25 and RRF, scikit-learn TF-IDF fallback, Streamlit, Ollama (llama3.2:3b) or the Anthropic API, Docker files.
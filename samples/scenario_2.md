# Disruption Impact Report

**Alert:** A truckers' strike is blocking bulky furniture and home appliance shipments out of Minas Gerais and Parana for the next 10 days

**Interpreted as:** origin = MG, PR, 10 days, scope = heavy, cause = a transport strike _(parsed by rules)_  
**Snapshot date:** 15 Jun 2018 | **LLM:** llama3.2:3b | **Retrieval:** chroma

## Executive summary

A transport strike in Minas Gerais and Parana for 10 days puts **69** active orders in scope. **34** will be delayed, but the SLA buffer absorbs 18 of them; **16** (23%) will miss their promised date, 9 of which are bulky/heavy goods, putting R$ 4,377 of order value at risk. Most exposed sellers: 0241d4d5 (Curitiba, 2 orders), 1554a685 (Monte Siao, 2 orders), a425f92c (Curitiba, 2 orders). Recommended: (1) proactively notify the 16 affected customers today with the new dates; (2) hold heavy freight at origin and reroute only light parcels (extra freight ~R$ 56); budget ~R$ 222 in goodwill vouchers.

## Key numbers

| Metric | Value |
|---|---|
| Active orders retrieved from origin | 69 |
| Actually delayed | 34 |
| Absorbed by SLA buffer (no action) | 18 |
| **Will miss promised date** | **16** |
| Heavy/bulky among breaching | 9 |
| Order value at risk | R$ 4,377 |
| Goodwill voucher budget | R$ 222 |
| Extra reroute freight | R$ 56 |

**Severity tiers:** Absorbed by SLA buffer: 18, Low: 9, Medium: 5, High: 2


**Actions:** NO_ACTION: 18, DELAY: 10, REROUTE: 5, CANCEL_OFFER: 1

## Most impacted sellers

| seller_id   | city           | state   |   orders |   heavy |   avg_severity |   value_brl |
|:------------|:---------------|:--------|---------:|--------:|---------------:|------------:|
| 0241d4d5    | curitiba       | PR      |        2 |       2 |           31   |       499.8 |
| 1554a685    | monte siao     | MG      |        2 |       0 |           17.5 |       315.8 |
| a425f92c    | curitiba       | PR      |        2 |       0 |           15.5 |       227   |
| 7fdb0720    | toledo         | PR      |        1 |       1 |           57   |       287   |
| 08d1c947    | belo horizonte | MG      |        1 |       1 |           55   |       538   |
| f62d7bb4    | vicosa         | MG      |        1 |       1 |           49   |       750   |
| ce7d1888    | baependi       | MG      |        1 |       1 |           47   |       138   |
| b3476778    | curitiba       | PR      |        1 |       1 |           39   |       449   |

## Most impacted categories

| categories                            |   orders |   avg_weight_kg |   avg_severity |
|:--------------------------------------|---------:|----------------:|---------------:|
| furniture_decor                       |        7 |             7   |           29.7 |
| housewares                            |        3 |            10   |           37.7 |
| furniture_living_room                 |        2 |             5.6 |           17.5 |
| furniture_bedroom                     |        1 |            10.8 |           57   |
| small_appliances                      |        1 |             3.1 |           27   |
| small_appliances_home_oven_and_coffee |        1 |             9   |           49   |
| sports_leisure                        |        1 |             9.4 |           23   |

## Highest-severity orders

| order_ref   | seller_state   | customer_state   |   total_weight_kg |   delay_days |   breach_days |   residual_breach |   severity | tier   | action       |   voucher_brl |
|:------------|:---------------|:-----------------|------------------:|-------------:|--------------:|------------------:|-----------:|:-------|:-------------|--------------:|
| e67dfbc4    | PR             | SC               |              10.8 |           13 |             7 |                 7 |         57 | High   | CANCEL_OFFER |          28.7 |
| dc072ac6    | MG             | SP               |              27.4 |           10 |             3 |                 3 |         55 | High   | DELAY        |          26.9 |
| bb05d405    | MG             | RJ               |               9   |           10 |             5 |                 5 |         49 | Medium | DELAY        |          50   |
| c47b0014    | MG             | ES               |              10.2 |           10 |             5 |                 5 |         47 | Medium | DELAY        |          13.8 |
| 893233a6    | PR             | SP               |              10.2 |           10 |             3 |                 3 |         39 | Medium | DELAY        |          22.4 |
| db655922    | PR             | MS               |               8.2 |           10 |             4 |                 4 |         38 | Medium | DELAY        |          25   |
| 841f2255    | PR             | MG               |               2.6 |           13 |             5 |                 0 |         38 | Medium | REROUTE      |           0   |
| 396a800e    | MG             | PR               |               9.7 |            9 |             1 |                 1 |         27 | Low    | DELAY        |          15.6 |

## Sample customer emails

### Order #e67dfbc4 - action `CANCEL_OFFER` (source: llm:llama3.2:3b)

**Subject:** Update on Order e67dfbc4

```
Dear valued customer,

We are writing to inform you that the new delivery date for your order e67dfbc4 is 12 Jul 2018. We apologize for any inconvenience caused by the transport strike that led to this delay.

Please note that a voucher of R$ 28.70 has already been added to your account, and no further action is required to claim it.

If you would like to request a full refund, please reply to this email. Please note that this is only for requesting a refund and not for any other purpose.

Sincerely,
Customer Care Team
```

## Agent trace (audit log)

1. **Disruption Parser** - states=['MG', 'PR'] cities=[] days=10 scope=heavy by=rules [] _(15 ms)_
2. **Retriever (hybrid RAG)** - 69 active orders in scope; backend=chroma; filter->dense+BM25->RRF _(587 ms)_
3. **Logistics Analyst** - 34 delayed, 16 breach SLA _(54 ms)_
4. **Mitigation Agent** - actions={'NO_ACTION': 18, 'DELAY': 10, 'REROUTE': 5, 'CANCEL_OFFER': 1} _(22 ms)_
5. **Impact Summarizer** - template narrative, 100 words _(3 ms)_
6. **Customer Success Agent** - 1 drafts; sources=['llm:llama3.2:3b']; guardrail issues=[[]] _(84875 ms)_
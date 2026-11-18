# Disruption Impact Report

**Alert:** Heavy rains are delaying all freight leaving Sao Paulo for the next 5 days

**Interpreted as:** origin = SP, 5 days, scope = all, cause = heavy rains _(parsed by rules)_  
**Snapshot date:** 15 Jun 2018 | **LLM:** llama3.2:3b | **Retrieval:** chroma

## Executive summary

Heavy rains in Sao Paulo for 5 days puts **1,543** active orders in scope. **581** will be delayed, but the SLA buffer absorbs 525 of them; **56** (4%) will miss their promised date, 7 of which are bulky/heavy goods, putting R$ 15,685 of order value at risk. Most exposed sellers: 2eb70248 (Campinas, 13 orders), d71d863e (Osasco, 6 orders), 7d13fca1 (Ribeirao Preto, 5 orders). Recommended: (1) proactively notify the 56 affected customers today with the new dates; (2) hold heavy freight at origin and reroute only light parcels (extra freight ~R$ 349); budget ~R$ 569 in goodwill vouchers.

## Key numbers

| Metric | Value |
|---|---|
| Active orders retrieved from origin | 1,543 |
| Actually delayed | 581 |
| Absorbed by SLA buffer (no action) | 525 |
| **Will miss promised date** | **56** |
| Heavy/bulky among breaching | 7 |
| Order value at risk | R$ 15,685 |
| Goodwill voucher budget | R$ 569 |
| Extra reroute freight | R$ 349 |

**Severity tiers:** Absorbed by SLA buffer: 525, Low: 25, Medium: 17, High: 14


**Actions:** NO_ACTION: 525, REROUTE: 45, DELAY: 7, CANCEL_OFFER: 4

## Most impacted sellers

| seller_id   | city                  | state   |   orders |   heavy |   avg_severity |   value_brl |
|:------------|:----------------------|:--------|---------:|--------:|---------------:|------------:|
| 2eb70248    | campinas              | SP      |       13 |       0 |           40.4 |      3593.2 |
| d71d863e    | osasco                | SP      |        6 |       0 |           45.3 |      2654   |
| 7d13fca1    | ribeirao preto        | SP      |        5 |       0 |           18   |      1189   |
| ad781527    | sao jose do rio preto | SP      |        4 |       4 |           53   |       560   |
| 1025f0e2    | sao paulo             | SP      |        2 |       0 |           36.5 |       255   |
| 23d7c96d    | capivari              | SP      |        1 |       1 |           50   |      1079.9 |
| c394e193    | sao paulo             | SP      |        1 |       0 |           47   |       285   |
| c60b801f    | guarulhos             | SP      |        1 |       0 |           46   |       218   |

## Most impacted categories

| categories                |   orders |   avg_weight_kg |   avg_severity |
|:--------------------------|---------:|----------------:|---------------:|
| watches_gifts             |       14 |             0.8 |           30.8 |
| telephony                 |        6 |             0.6 |           42.8 |
| computers_accessories     |        5 |             0.6 |           37.8 |
| health_beauty             |        5 |             1.5 |           36.4 |
| baby                      |        4 |            10.7 |           53   |
| construction_tools_safety |        3 |             1.2 |           31   |
| furniture_decor           |        2 |             2   |           32   |
| bed_bath_table            |        2 |             1.2 |           31   |

## Highest-severity orders

| order_ref   | seller_state   | customer_state   |   total_weight_kg |   delay_days |   breach_days |   residual_breach |   severity | tier   | action       |   voucher_brl |
|:------------|:---------------|:-----------------|------------------:|-------------:|--------------:|------------------:|-----------:|:-------|:-------------|--------------:|
| 52b828f1    | SP             | SP               |              10.7 |            7 |             7 |                 7 |         53 | High   | CANCEL_OFFER |          14   |
| 57c432fd    | SP             | SP               |              10.7 |            7 |             7 |                 7 |         53 | High   | CANCEL_OFFER |          14   |
| eb3c78fe    | SP             | SP               |              10.7 |            7 |             7 |                 7 |         53 | High   | CANCEL_OFFER |          14   |
| 8124e0a6    | SP             | SP               |              10.7 |            7 |             7 |                 7 |         53 | High   | CANCEL_OFFER |          14   |
| 843ad442    | SP             | AL               |               3.8 |            7 |             7 |                 2 |         53 | High   | REROUTE      |           7.5 |
| 10dc93d4    | SP             | SP               |               0.6 |            7 |             7 |                 2 |         51 | High   | REROUTE      |          32.4 |
| 6c54ede9    | SP             | SP               |               0.8 |            7 |             7 |                 2 |         51 | High   | REROUTE      |          24.7 |
| 9d8422be    | SP             | SP               |               0.6 |            7 |             7 |                 2 |         51 | High   | REROUTE      |          32.4 |

## Sample customer emails

### Order #52b828f1 - action `CANCEL_OFFER` (source: llm:llama3.2:3b)

**Subject:** Update on Order #52b828f1

```
Dear Valued Customer,

We are writing to inform you that the new delivery date for your order #52b828f1 is now expected to be 23 Jun 2018.

We apologise for any inconvenience caused by the disruption in our service due to heavy rains in Sao Paulo.

As a gesture of goodwill, we would like to offer you a voucher of R$ 14.00, which has already been added to your account.

Please note that replying to this email is only to request a full refund for your order. No further action is required to claim the voucher.

We acknowledge that your order was already running late before the disruption, and this update adds further delay to the original ETA of 1 Jun 2018.

Thank you for your patience and understanding.

Customer Care Team
```

## Agent trace (audit log)

1. **Disruption Parser** - states=['SP'] cities=[] days=5 scope=all by=rules ["'Sao Paulo' interpreted as the state (SP) - say 'city of Sao Paulo' to narrow"] _(13 ms)_
2. **Retriever (hybrid RAG)** - 1543 active orders in scope; backend=chroma; filter->dense+BM25->RRF _(644 ms)_
3. **Logistics Analyst** - 581 delayed, 56 breach SLA _(94 ms)_
4. **Mitigation Agent** - actions={'NO_ACTION': 525, 'REROUTE': 45, 'DELAY': 7, 'CANCEL_OFFER': 4} _(30 ms)_
5. **Impact Summarizer** - template narrative, 97 words _(3 ms)_
6. **Customer Success Agent** - 1 drafts; sources=['llm:llama3.2:3b']; guardrail issues=[[]] _(102054 ms)_
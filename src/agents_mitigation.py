"""Agent 3 - Mitigation Agent: a transparent policy engine (rules, not LLM).
Money and promises to customers must be deterministic, explainable and auditable.
The LLM is deliberately NOT allowed to decide refunds, vouchers or dates."""
import pandas as pd
from . import config as C


def _voucher_pct(breach):
    for cap, pct in C.VOUCHER_TIERS:
        if breach <= cap: return pct
    return 0.0


def decide(r) -> dict:
    heavy = bool(r.heavy)
    # 1. nothing to do if the SLA buffer absorbs the delay
    if r.breach_days == 0:
        return dict(action="NO_ACTION", residual_breach=0, new_eta=r.order_estimated_delivery_date.normalize(),
                    voucher_brl=0.0, extra_cost_brl=0.0,
                    rationale=f"Delay of {r.delay_days}d is absorbed by the SLA buffer ({r.sla_buffer_days:.0f}d). Monitor only.")
    # 2. reroute: only for parcels still at the seller, light, and delayed long enough to justify extra cost
    if (r.status_asof == "awaiting_dispatch" and not heavy and r.total_weight_kg <= C.REROUTE_MAX_KG
            and r.delay_days >= C.REROUTE_MIN_DELAY_DAYS):
        eta = min(r.projected_eta, r.baseline_eta + pd.Timedelta(days=C.REROUTE_LAG_DAYS))
        resid = max(0, (eta - max(r.order_estimated_delivery_date, r.baseline_eta)).days)
        extra = round(r.freight_value * C.REROUTE_COST_UPLIFT, 2)
        pct = _voucher_pct(resid) if resid > 0 else 0
        return dict(action="REROUTE", residual_breach=resid, new_eta=eta, voucher_brl=round(min(r.items_value * pct, C.VOUCHER_CAP_BRL), 2),
                    extra_cost_brl=extra, rationale=f"Light parcel ({r.total_weight_kg:.1f} kg) still at seller: reroute via alternate hub "
                    f"(+R${extra:.2f} freight, +{C.REROUTE_LAG_DAYS}d vs. normal) cuts delay {r.delay_days}d -> {resid}d beyond SLA.")
    # 3. otherwise the item waits. Long residual breach => give the customer the option to cancel (never auto-cancel)
    resid = r.breach_days
    pct = _voucher_pct(resid)
    voucher = round(min(r.items_value * pct, C.VOUCHER_CAP_BRL), 2)
    can_cancel = r.status_asof == "awaiting_dispatch"
    why_not = ("heavy/bulky freight: alternate-hub rerouting costs more than the delay it saves" if heavy
               else "parcel already in carrier network: cannot be pulled back" if r.status_asof == "in_transit"
               else "delay too short to justify reroute cost")
    if resid >= C.CANCEL_OFFER_BREACH_DAYS and can_cancel:
        return dict(action="CANCEL_OFFER", residual_breach=resid, new_eta=r.projected_eta, voucher_brl=voucher, extra_cost_brl=0.0,
                    rationale=f"{resid}d past SLA ({why_not}). Offer full refund option or wait with {int(pct*100)}% voucher.")
    return dict(action="DELAY", residual_breach=resid, new_eta=r.projected_eta, voucher_brl=voucher, extra_cost_brl=0.0,
                rationale=f"Hold & notify: {why_not}. {resid}d past SLA -> {int(pct*100)}% goodwill voucher (cap R${C.VOUCHER_CAP_BRL:.0f}).")


def mitigate(impacted: pd.DataFrame) -> pd.DataFrame:
    if impacted.empty: return impacted
    res = pd.DataFrame([decide(r) for r in impacted.itertuples()], index=impacted.index)
    return pd.concat([impacted, res], axis=1)

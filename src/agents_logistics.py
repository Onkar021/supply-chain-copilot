"""Agent 2 - Logistics Analyst. 100% deterministic: numbers must be reproducible and auditable.

For each retrieved order we answer: (1) is it actually held up? (2) by how many days?
(3) does that breach the promised delivery date (SLA)? (4) how severe is it?"""
import numpy as np, pandas as pd
from . import config as C


def _days(td): return np.ceil(td.dt.total_seconds() / 86400)


def analyse(orders: pd.DataFrame, disruption, buffers, as_of=C.AS_OF) -> pd.DataFrame:
    dur = disruption.duration_days
    end = as_of + pd.Timedelta(days=dur)
    df = orders.copy()
    recov = int(np.ceil(C.RECOVERY_FACTOR * dur))

    # (1)+(2) delay: parcels still at the seller wait until the window closes + backlog recovery;
    #         parcels just picked up are partially exposed; everything else already left the origin network.
    dispatch = df.shipping_limit.fillna(as_of).clip(lower=as_of)
    hold = (_days(end - dispatch) + recov).clip(lower=0)
    hold = np.where(dispatch > end, 0, hold)
    awaiting = df.status_asof == "awaiting_dispatch"
    exposed = (df.status_asof == "in_transit") & \
              (df.order_delivered_carrier_date >= as_of - pd.Timedelta(days=C.IN_TRANSIT_EXPOSURE_DAYS))
    df["delay_days"] = np.where(awaiting, hold, np.where(exposed, np.ceil(C.IN_TRANSIT_DELAY_FACTOR * dur), 0)).astype(int)

    # (3) SLA: promised date minus the slack orders historically have (learned from data, conservative percentile)
    buf = df.customer_state.map(buffers["by_state"]).fillna(buffers["global"])
    df["sla_buffer_days"] = buf
    df["baseline_eta"] = (df.order_estimated_delivery_date - pd.to_timedelta(buf, unit="D")) \
        .clip(lower=as_of + pd.Timedelta(days=1)).dt.normalize()
    df["projected_eta"] = df.baseline_eta + pd.to_timedelta(df.delay_days, unit="D")
    breach = _days(df.projected_eta - df.order_estimated_delivery_date).clip(lower=0)
    # df["breach_days"] = np.where(df.delay_days > 0, breach, 0).astype(int)
    df["already_late"] = df.order_estimated_delivery_date < as_of
    df["breach_days"] = np.where(df.delay_days > 0, np.minimum(breach, df.delay_days), 0).astype(int)

    # (4) severity 0-100 = SLA breach (50) + weight (25) + order value at stake (15) + freight burden (10)
    freight_ratio = df.freight_value / df.items_value.clip(lower=1)
    comp = pd.DataFrame({"breach": np.clip(df.breach_days / 10, 0, 1) * 50,
                         "weight": np.clip(df.total_weight_kg / 30, 0, 1) * 25,
                         "value": np.clip(df.items_value / 500, 0, 1) * 15,
                         "freight": np.clip(freight_ratio / .5, 0, 1) * 10})
    sev = comp.sum(axis=1).round().astype(int)
    df["severity"] = np.where(df.delay_days == 0, 0, np.where(df.breach_days == 0, np.minimum(sev, 15), sev))
    df["tier"] = np.select([df.delay_days == 0, df.breach_days == 0, df.severity >= 70, df.severity >= 50, df.severity >= 30],
                           ["Not impacted", "Absorbed by SLA buffer", "Critical", "High", "Medium"], "Low")
    df["severity_why"] = comp.round(0).astype(int).apply(
        lambda r: f"breach {r.breach}/50 + weight {r.weight}/25 + value {r.value}/15 + freight {r.freight}/10", axis=1)
    return df

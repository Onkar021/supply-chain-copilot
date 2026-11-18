"""Step 1 - Data layer: build ONE row per order from the Olist tables, flag what is 'active' at the
simulated snapshot date, and learn SLA padding from history (no hard-coded guesses)."""
import numpy as np, pandas as pd
from . import config as C

REQUIRED = ["olist_orders_dataset.csv", "olist_order_items_dataset.csv", "olist_products_dataset.csv",
            "olist_sellers_dataset.csv", "olist_customers_dataset.csv"]
DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"]


def _read(name, **kw):
    return pd.read_csv(C.RAW_DIR / name, **kw)


def ensure_raw():
    if all((C.RAW_DIR / f).exists() for f in REQUIRED):
        return "olist"
    from .synthetic import generate
    print("[data] Olist CSVs not found in data/raw -> generating synthetic Olist-shaped data (demo mode)")
    generate(C.RAW_DIR)
    return "synthetic"


def build_corpus():
    """Returns (orders_df, sla_buffers). Cached as pickle for fast restarts."""
    cache = C.PROC_DIR / "corpus.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    src = ensure_raw()
    orders = _read("olist_orders_dataset.csv", parse_dates=DATES)
    items = _read("olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    prods = _read("olist_products_dataset.csv")
    sellers = _read("olist_sellers_dataset.csv")
    cust = _read("olist_customers_dataset.csv")
    tr = C.RAW_DIR / "product_category_name_translation.csv"
    cmap = dict(pd.read_csv(tr).values) if tr.exists() else {}

    # product cleaning: English category, impute missing weight/volume by category median
    prods["category"] = prods.product_category_name.map(cmap).fillna(prods.product_category_name).fillna("unknown")
    prods["vol"] = prods.product_length_cm * prods.product_height_cm * prods.product_width_cm
    for col in ["product_weight_g", "vol"]:
        prods[col] = prods.groupby("category")[col].transform(lambda s: s.fillna(s.median())).fillna(prods[col].median())
    prods["w_kg"] = prods.product_weight_g / 1000

    items = items.merge(prods[["product_id", "category", "w_kg", "vol"]], on="product_id", how="left") \
                 .merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left")
    g = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), total_weight_kg=("w_kg", "sum"), total_volume_cm3=("vol", "sum"),
        items_value=("price", "sum"), freight_value=("freight_value", "sum"), shipping_limit=("shipping_limit_date", "max"),
        seller_id=("seller_id", "first"), seller_city=("seller_city", "first"), seller_state=("seller_state", "first"),
        categories=("category", lambda s: sorted(set(s)))).reset_index()
    df = orders.merge(g, on="order_id").merge(cust[["customer_id", "customer_city", "customer_state"]], on="customer_id")

    # SLA padding learned from history: how many days earlier than promised do orders really arrive?
    done = df[df.order_delivered_customer_date.notna() & (df.order_delivered_customer_date <= C.AS_OF)]
    pad = (done.order_estimated_delivery_date - done.order_delivered_customer_date).dt.days
    q = C.SLA_BUFFER_PERCENTILE
    buffers = {"global": float(np.clip(pad.quantile(q), 0, 20)),
               "by_state": pad.groupby(done.customer_state).quantile(q).clip(0, 20).to_dict()}

    # corpus = recent orders; 'active' = not delivered/cancelled at snapshot
    df = df[(df.order_purchase_timestamp <= C.AS_OF) &
            (df.order_purchase_timestamp > C.AS_OF - pd.Timedelta(days=C.CORPUS_WINDOW_DAYS))].copy()
    df["is_active"] = (~df.order_status.isin(["canceled", "unavailable"])) & \
                      (df.order_delivered_customer_date.isna() | (df.order_delivered_customer_date > C.AS_OF))
    sent = df.order_delivered_carrier_date.notna() & (df.order_delivered_carrier_date <= C.AS_OF)
    df["status_asof"] = np.where(~df.is_active, "closed", np.where(sent, "in_transit", "awaiting_dispatch"))
    df["heavy"] = (df.total_weight_kg >= C.HEAVY_KG) | (df.total_volume_cm3 >= C.HEAVY_CM3)
    df["order_ref"] = df.order_id.str[:8]
    df = df.sort_values(["is_active", "order_purchase_timestamp"], ascending=False).head(C.MAX_ORDERS).reset_index(drop=True)
    for c in ["seller_city", "customer_city"]:
        df[c] = df[c].fillna("unknown")
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((df, buffers), cache)
    print(f"[data] source={src} corpus={len(df)} orders, active={int(df.is_active.sum())}, "
          f"heavy={int(df.heavy.sum())}, SLA buffer(global p{int(q*100)})={buffers['global']:.1f}d")
    return df, buffers

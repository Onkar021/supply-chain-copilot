"""Generates an Olist-shaped synthetic dataset so the project runs without a Kaggle download.
Column names match the real Olist CSVs, so the real files are a drop-in replacement."""
import numpy as np, pandas as pd
from pathlib import Path

CATS = [  # pt name, en name, mean weight g, share
    ("moveis_decoracao", "furniture_decor", 6000, .10), ("moveis_sala", "furniture_living_room", 15000, .03),
    ("moveis_escritorio", "office_furniture", 20000, .03), ("cama_mesa_banho", "bed_bath_table", 1500, .12),
    ("eletrodomesticos", "home_appliances", 9000, .05), ("utilidades_domesticas", "housewares", 2500, .08),
    ("beleza_saude", "health_beauty", 400, .12), ("esporte_lazer", "sports_leisure", 1200, .10),
    ("informatica_acessorios", "computers_accessories", 500, .10), ("brinquedos", "toys", 900, .07),
    ("relogios_presentes", "watches_gifts", 300, .08), ("telefonia", "telephony", 200, .06),
    ("ferramentas_jardim", "garden_tools", 3500, .06)]
SELLER_CITIES = {"SP": ["sao paulo"] * 5 + ["campinas", "santos", "guarulhos", "ribeirao preto", "sorocaba"],
                 "RJ": ["rio de janeiro", "niteroi"], "MG": ["belo horizonte", "uberlandia"],
                 "PR": ["curitiba", "londrina"], "SC": ["florianopolis", "joinville"],
                 "RS": ["porto alegre", "caxias do sul"], "BA": ["salvador"], "GO": ["goiania"], "DF": ["brasilia"]}
S_W = {"SP": .58, "RJ": .10, "MG": .10, "PR": .07, "SC": .05, "RS": .04, "BA": .02, "GO": .02, "DF": .02}
C_W = {"SP": .42, "RJ": .13, "MG": .12, "RS": .055, "PR": .05, "SC": .036, "BA": .034, "DF": .025,
       "GO": .02, "ES": .02, "PE": .017, "CE": .013, "PA": .01}
FMT = "%Y-%m-%d %H:%M:%S"


def generate(out: Path, n_orders=20000, seed=7):
    rng = np.random.default_rng(seed); out.mkdir(parents=True, exist_ok=True)
    pick = lambda w, n: rng.choice(list(w), n, p=np.array(list(w.values())) / sum(w.values()))
    n_s = 600; ss = pick(S_W, n_s)
    sellers = pd.DataFrame({"seller_id": [f"{i:08x}{rng.integers(16**8):08x}{'a'*16}" for i in range(n_s)],
                            "seller_zip_code_prefix": rng.integers(1000, 99999, n_s), "seller_state": ss,
                            "seller_city": [rng.choice(SELLER_CITIES[s]) for s in ss]})
    n_p = 3000; ci = rng.choice(len(CATS), n_p, p=np.array([c[3] for c in CATS]) / sum(c[3] for c in CATS))
    w = np.array([CATS[i][2] for i in ci]) * rng.lognormal(0, .5, n_p)
    side = (np.clip(w, 100, 40000) ** (1 / 3)) * 2.2
    products = pd.DataFrame({"product_id": [f"p{i:06x}{'b'*24}" for i in range(n_p)],
                             "product_category_name": [CATS[i][0] for i in ci], "product_weight_g": w.round(),
                             "product_length_cm": (side * rng.uniform(.8, 1.4, n_p)).round(),
                             "product_height_cm": (side * rng.uniform(.5, 1.1, n_p)).round(),
                             "product_width_cm": (side * rng.uniform(.6, 1.2, n_p)).round()})
    products.loc[rng.random(n_p) < .02, "product_weight_g"] = np.nan   # real data has gaps
    trans = pd.DataFrame({"product_category_name": [c[0] for c in CATS], "product_category_name_english": [c[1] for c in CATS]})
    cstate = pick(C_W, n_orders)
    customers = pd.DataFrame({"customer_id": [f"c{i:07x}{'c'*24}" for i in range(n_orders)],
                              "customer_unique_id": [f"u{i:07x}{'d'*24}" for i in range(n_orders)],
                              "customer_zip_code_prefix": rng.integers(1000, 99999, n_orders),
                              "customer_city": [f"city_{s.lower()}_{rng.integers(1, 6)}" for s in cstate], "customer_state": cstate})
    purchase = pd.Timestamp("2018-01-15") + pd.to_timedelta(rng.uniform(0, 166, n_orders), unit="D")
    approved = purchase + pd.to_timedelta(rng.exponential(6, n_orders), unit="h")
    carrier = approved + pd.to_timedelta(rng.gamma(2, 1.2, n_orders), unit="D")
    oid = np.array([f"o{i:07x}{'e'*24}" for i in range(n_orders)])
    n_items = rng.choice([1, 2, 3], n_orders, p=[.85, .12, .03])
    seller_idx = rng.integers(0, n_s, n_orders)
    o_seller = sellers.seller_state.values[seller_idx]
    transit = np.where(o_seller == cstate, 3, rng.uniform(5, 11, n_orders)) * rng.uniform(.8, 1.3, n_orders)
    delivered = carrier + pd.to_timedelta(transit, unit="D")
    est = purchase.normalize() + pd.to_timedelta(np.clip(np.round((delivered - purchase).total_seconds() / 86400 + rng.normal(9, 4, n_orders)), 5, None), unit="D")
    status = np.where(delivered <= pd.Timestamp("2018-08-01"), "delivered", "shipped")
    canc = rng.random(n_orders) < .015; status = np.where(canc, "canceled", status)
    orders = pd.DataFrame({"order_id": oid, "customer_id": customers.customer_id, "order_status": status,
                           "order_purchase_timestamp": purchase, "order_approved_at": approved,
                           "order_delivered_carrier_date": carrier, "order_delivered_customer_date": delivered,
                           "order_estimated_delivery_date": est})
    orders.loc[canc, ["order_delivered_carrier_date", "order_delivered_customer_date"]] = pd.NaT
    orders.loc[status == "shipped", "order_delivered_customer_date"] = pd.NaT
    rows = np.repeat(np.arange(n_orders), n_items)
    pid = rng.integers(0, n_p, len(rows)); pr = products.iloc[pid]
    wk = pr.product_weight_g.fillna(1000).values / 1000
    price = np.round(rng.lognormal(4.2, .8, len(rows)), 2)
    freight = np.round(7 + wk * 1.6 + rng.uniform(0, 8, len(rows)), 2)
    items = pd.DataFrame({"order_id": oid[rows], "order_item_id": np.concatenate([np.arange(1, k + 1) for k in n_items]),
                          "product_id": pr.product_id.values, "seller_id": sellers.seller_id.values[seller_idx[rows]],
                          "shipping_limit_date": approved.values[rows] + pd.to_timedelta(rng.uniform(2, 4, len(rows)), unit="D"),
                          "price": price, "freight_value": freight})
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c]).dt.strftime(FMT)
    items["shipping_limit_date"] = pd.to_datetime(items.shipping_limit_date).dt.strftime(FMT)
    for name, d in [("olist_orders_dataset", orders), ("olist_order_items_dataset", items), ("olist_products_dataset", products),
                    ("olist_sellers_dataset", sellers), ("olist_customers_dataset", customers),
                    ("product_category_name_translation", trans)]:
        d.to_csv(out / f"{name}.csv", index=False)

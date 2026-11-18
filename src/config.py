"""Central config. Every business assumption lives here so it is easy to explain and tune."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
OUT_DIR = ROOT / "outputs"

# ---- Simulation clock -------------------------------------------------------
# Olist is historical, so we "replay" it: pick a snapshot date and treat orders
# that are not yet delivered on that date as the live, active book of orders.
AS_OF = pd.Timestamp(os.getenv("AS_OF_DATE", "2018-06-15"))
CORPUS_WINDOW_DAYS = 45            # orders purchased in the 45 days before AS_OF
MAX_ORDERS = int(os.getenv("MAX_ORDERS", 10000))

# ---- Models -----------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ---- Logistics assumptions (documented in README) --------------------------
RECOVERY_FACTOR = 0.25             # backlog clearing adds 25% of disruption length
IN_TRANSIT_EXPOSURE_DAYS = 2       # parcels picked up <=2 days ago are still in origin network
IN_TRANSIT_DELAY_FACTOR = 0.5      # those parcels lose ~half the disruption length
HEAVY_KG = 8.0                     # bulky/heavy if total weight >= 8 kg ...
HEAVY_CM3 = 60_000                 # ... or volume >= 60 L (~40x40x40 cm)
SLA_BUFFER_PERCENTILE = 0.25       # conservative: only count the slack 75% of orders really have

# ---- Mitigation policy ------------------------------------------------------
REROUTE_MAX_KG = 5.0               # only light parcels are cost-effective to reroute
REROUTE_MIN_DELAY_DAYS = 3
REROUTE_COST_UPLIFT = 0.35         # +35% freight via alternate hub
REROUTE_LAG_DAYS = 2               # extra days an alternate route adds vs. normal ETA
CANCEL_OFFER_BREACH_DAYS = 7       # offer refund option when residual SLA breach >= 7 days
VOUCHER_TIERS = [(3, 0.05), (7, 0.10), (10**6, 0.15)]   # (max breach days, % of item value)
VOUCHER_CAP_BRL = 50.0

# ---- Retrieval --------------------------------------------------------------
RRF_K = 60
HEAVY_CATEGORY_HINTS = ["furniture", "appliance", "housewares", "bed_bath", "home_",
                        "air_conditioning", "garden_tools", "construction", "mattress"]

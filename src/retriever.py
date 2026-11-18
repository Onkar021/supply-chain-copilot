"""Step 3 - RAG layer: hybrid retrieval over order documents.

  hard metadata filter (geography + active)  -> candidate set, 100% recall, no embedding error
  dense semantic ranking (ChromaDB + MiniLM) -> 'what is closest in meaning to this disruption'
  sparse BM25 ranking                        -> exact terms (city names, categories, 'bulky')
  Reciprocal Rank Fusion                     -> one robust ranking, no score calibration needed

Falls back to TF-IDF as the dense component if chromadb/sentence-transformers are not installed."""
import math, os, re
import numpy as np, pandas as pd
from . import config as C

_TOK = re.compile(r"[a-z0-9_]+")
tok = lambda s: _TOK.findall(s.lower())


def order_to_doc(r) -> str:
    """Turn a structured order into a natural-language document that embeds well."""
    cats = ", ".join(c.replace("_", " ") for c in r.categories)
    return (f"Order {r.order_ref}. Seller {r.seller_id[:8]} located in {r.seller_city.title()}, {r.seller_state}. "
            f"Ships to {str(r.customer_city).title()}, {r.customer_state}. Products: {cats}. "
            f"{r.n_items} item(s), {r.total_weight_kg:.1f} kg. {'Bulky heavy freight.' if r.heavy else 'Light parcel.'} "
            f"Status: {r.status_asof.replace('_', ' ')}. Promised delivery {r.order_estimated_delivery_date:%d %b %Y}.")


def bm25(query_tokens, docs_tokens, k1=1.5, b=0.75):
    N = len(docs_tokens); avgdl = (sum(len(d) for d in docs_tokens) / N) or 1
    dfc = {}
    for d in docs_tokens:
        for t in set(d):
            dfc[t] = dfc.get(t, 0) + 1
    out = np.zeros(N)
    for i, d in enumerate(docs_tokens):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        for q in set(query_tokens):
            if q in tf:
                idf = math.log(1 + (N - dfc[q] + .5) / (dfc[q] + .5))
                out[i] += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avgdl))
    return out


class OrderIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.set_index("order_id", drop=False)
        self.docs = {r.order_id: order_to_doc(r) for r in df.itertuples()}
        self.backend = "tfidf"
        if os.getenv("FORCE_TFIDF") != "1":
            try:
                self._init_chroma(df); self.backend = "chroma"
            except Exception as e:                      # noqa
                print(f"[retriever] Chroma unavailable ({type(e).__name__}: {e}) -> TF-IDF fallback")
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.mat = self.vec.fit_transform([self.docs[i] for i in self.df.index])
            self.pos = {oid: i for i, oid in enumerate(self.df.index)}

    # ---------------- chroma ----------------
    def _init_chroma(self, df):
        import chromadb
        from chromadb.utils import embedding_functions
        C.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=C.EMBED_MODEL)
        col = client.get_or_create_collection("orders", embedding_function=ef, metadata={"hnsw:space": "cosine"})
        if col.count() != len(df):
            print(f"[retriever] embedding {len(df)} orders with {C.EMBED_MODEL} (one-off, ~1-3 min on CPU)...")
            client.delete_collection("orders")
            col = client.create_collection("orders", embedding_function=ef, metadata={"hnsw:space": "cosine"})
            for s in range(0, len(df), 500):
                ch = df.iloc[s:s + 500]
                col.add(ids=ch.order_id.tolist(), documents=[self.docs[i] for i in ch.order_id],
                        metadatas=[{"seller_state": r.seller_state, "seller_city": r.seller_city,
                                    "status": r.status_asof, "is_active": bool(r.is_active), "heavy": bool(r.heavy)}
                                   for r in ch.itertuples()])
        self.col = col

    @staticmethod
    def _where(states, cities, active_only):
        cl = []
        if active_only: cl.append({"is_active": True})
        if states: cl.append({"seller_state": {"$in": list(states)}})
        if cities: cl.append({"seller_city": {"$in": list(cities)}})
        return None if not cl else cl[0] if len(cl) == 1 else {"$and": cl}

    def _candidates(self, states, cities, active_only):
        if self.backend == "chroma":
            r = self.col.get(where=self._where(states, cities, active_only), include=["documents"])
            return r["ids"], r["documents"]
        m = pd.Series(True, index=self.df.index)
        if active_only: m &= self.df.is_active
        if states: m &= self.df.seller_state.isin(states)
        if cities: m &= self.df.seller_city.isin(cities)
        ids = self.df.index[m].tolist()
        return ids, [self.docs[i] for i in ids]

    def _dense_rank(self, query, ids, states, cities, active_only):
        if self.backend == "chroma":
            r = self.col.query(query_texts=[query], n_results=len(ids), include=["distances"],
                               where=self._where(states, cities, active_only))
            return dict(zip(r["ids"][0], [1 - d for d in r["distances"][0]]))
        from sklearn.metrics.pairwise import linear_kernel
        sims = linear_kernel(self.vec.transform([query]), self.mat[[self.pos[i] for i in ids]]).ravel()
        return dict(zip(ids, sims))

    # ---------------- public ----------------
    def hybrid_search(self, query, states=None, cities=None, active_only=True) -> pd.DataFrame:
        ids, docs = self._candidates(states, cities, active_only)
        if not ids:
            return pd.DataFrame()
        dense = self._dense_rank(query, ids, states, cities, active_only)
        dense_s = np.array([dense.get(i, 0) for i in ids])
        bm_s = bm25(tok(query), [tok(d) for d in docs])
        rank = lambda s: pd.Series(s).rank(ascending=False, method="first").values
        d_rank, b_rank = rank(dense_s), rank(bm_s)
        rrf = 1 / (C.RRF_K + d_rank) + 1 / (C.RRF_K + b_rank)
        out = self.df.loc[ids].copy()
        out["dense_score"], out["bm25_score"] = dense_s.round(3), bm_s.round(2)
        out["dense_rank"], out["bm25_rank"], out["rrf"] = d_rank.astype(int), b_rank.astype(int), rrf
        return out.sort_values("rrf", ascending=False)


KEYWORD_HINTS = {"furniture": ["furniture"], "appliance": ["appliance"], "fridge": ["appliance"],
                 "sofa": ["furniture_living"], "mattress": ["bed_bath", "mattress"],
                 "home goods": ["housewares", "bed_bath", "home_"], "bulky": [], "oversize": []}


def apply_scope(hits: pd.DataFrame, scope: str, keywords) -> pd.DataFrame:
    """'heavy' scope keeps bulky items (weight/volume rule) OR categories the alert actually names.
    Generic hints are used only when the alert names no specific product type."""
    if scope != "heavy" or hits.empty:
        return hits
    specific = [h for k in (keywords or []) for h in KEYWORD_HINTS.get(k, [])]
    hints = specific or C.HEAVY_CATEGORY_HINTS
    cat_match = hits.categories.apply(lambda cs: any(h in c for c in cs for h in hints))
    return hits[hits.heavy | cat_match]

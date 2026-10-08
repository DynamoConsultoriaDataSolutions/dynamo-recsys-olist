"""Recomendador híbrido item-to-item.

Idea: "quienes compraron este producto también compraron…", con dos respaldos
porque en Olist casi no hay co-compras (solo ~24% de los productos tiene alguna).

Para un usuario CON historial, se arma la lista en 3 niveles (sin repetir y sin
productos que ya compró), tomando como "semillas" sus compras más recientes:

  1. Co-compra: productos comprados por los mismos clientes que compraron la
     semilla, ordenados por similitud coseno.
  2. Contenido: productos de la misma categoría que la semilla, ordenados por
     popularidad (opcional: priorizar precio parecido con `price_tolerance`).
  3. Popularidad reciente (últimos `recent_days` días).

Para un usuario SIN historial (cold start) se usa directamente el nivel 3.

Similitud coseno entre productos i y j:
    clientes que compraron i y j / sqrt(clientes que compraron i * clientes que compraron j)
Vale 1 si siempre se compran juntos y 0 si nunca.

Hiperparámetros elegidos en VALIDACIÓN (nunca con el test), ver README:
  - recent_days=30: mejor ventana de popularidad en dos periodos de validación.
  - price_tolerance=None: filtrar por precio parecido empeoró el acierto en validación.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.popularity import PopularityRecommender, rank_desc


class ItemToItemRecommender:
    def __init__(self, recent_days=30, max_seeds=5, n_neighbors=50,
                 use_content=True, price_tolerance=None):
        self.recent_days = recent_days
        self.max_seeds = max_seeds
        self.n_neighbors = n_neighbors
        self.use_content = use_content
        self.price_tolerance = price_tolerance

    # ------------------------------------------------------------------ fit
    def fit(self, train):
        self.fallback_ = PopularityRecommender(recent_days=self.recent_days).fit(train)

        # Historial por usuario, de la compra más reciente a la más antigua
        hist = (train.sort_values(["purchase_ts", "product_id"], ascending=[False, True])
                .drop_duplicates(["customer_unique_id", "product_id"]))
        self.history_ = hist.groupby("customer_unique_id", sort=False)["product_id"].agg(list).to_dict()

        self.neighbors_ = self._co_purchase_neighbors(hist)

        # Atributos de producto para el nivel de contenido
        prod = train.groupby("product_id").agg(
            category=("category", "first"), price=("price", "median"),
            buyers=("customer_unique_id", "nunique"))
        self.category_ = prod["category"].to_dict()
        self.price_ = prod["price"].to_dict()
        self.by_category_ = {c: list(rank_desc(g["buyers"]).index)
                             for c, g in prod.groupby("category")}
        return self

    def _co_purchase_neighbors(self, hist):
        """dict producto -> [(vecino, similitud), ...] ordenado de mayor a menor."""
        n_prod = hist.groupby("customer_unique_id")["product_id"].transform("size")
        multi = hist[n_prod >= 2]                      # solo usuarios con 2+ productos aportan pares
        if multi.empty:
            return {}
        users = multi["customer_unique_id"].astype("category")
        items = multi["product_id"].astype("category")
        X = sparse.csr_matrix((np.ones(len(multi)), (users.cat.codes, items.cat.codes)))
        co = (X.T @ X).tocoo()                         # co[i, j] = clientes que compraron i y j
        counts = np.asarray(X.sum(axis=0)).ravel()
        mask = co.row != co.col
        i, j, c = co.row[mask], co.col[mask], co.data[mask]
        sim = c / np.sqrt(counts[i] * counts[j])
        ids = items.cat.categories
        pairs = pd.DataFrame({"item": ids[i], "neighbor": ids[j], "sim": sim})
        pairs = pairs.sort_values(["item", "sim", "neighbor"], ascending=[True, False, True])
        pairs = pairs.groupby("item").head(self.n_neighbors)
        return {it: list(zip(g["neighbor"], g["sim"])) for it, g in pairs.groupby("item")}

    # ------------------------------------------------------------ recommend
    def recommend(self, user_id, k=10):
        history = self.history_.get(user_id)
        if not history:
            return self.fallback_.recommend(user_id, k)       # cold start
        seen, seeds = set(history), history[: self.max_seeds]
        recs, taken = [], set(seen)

        def add(candidates):
            for p in candidates:
                if p not in taken:
                    recs.append(p)
                    taken.add(p)
                    if len(recs) >= k:
                        return True
            return False

        # Nivel 1: co-compra (similitudes sumadas entre semillas)
        scores = defaultdict(float)
        for s in seeds:
            for nb, sim in self.neighbors_.get(s, []):
                scores[nb] += sim
        if add(sorted(scores, key=lambda p: (-scores[p], p))):
            return recs

        # Nivel 2: contenido (misma categoría, precio parecido primero)
        if self.use_content:
            for s in seeds:
                same_cat = self.by_category_.get(self.category_.get(s), [])[: 200]
                p0 = self.price_.get(s)
                if self.price_tolerance and p0:
                    lo, hi = p0 / (1 + self.price_tolerance), p0 * (1 + self.price_tolerance)
                    if add([p for p in same_cat if lo <= self.price_.get(p, -1) <= hi]):
                        return recs
                if add(same_cat):
                    return recs

        # Nivel 3: popularidad reciente
        add(self.fallback_.recommend(user_id, k + len(taken)))
        return recs[:k]

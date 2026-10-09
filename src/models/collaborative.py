"""Modelo colaborativo por factorización de matrices (SVD) con feedback implícito.

Idea: se arma la matriz usuario-producto (compra = 1) y se factoriza con
TruncatedSVD en `n_factors` factores latentes. El puntaje de un producto para un
usuario es el producto punto entre el factor del usuario y el del producto, y se
recomiendan los de mayor puntaje que el usuario aún no compró.

Se complementa con el item-to-item de src/models/item_based.py: aquel se apoya en
co-compras directas; éste aprende patrones latentes. Cumple el protocolo común
(src/evaluation/evaluate.py):
    model.fit(train_df)
    model.recommend(user_id, k) -> lista de product_id sin repetir

Cold start: si el usuario no tiene historial en train (~97% de los usuarios de
test) o el modelo no completa K recomendaciones, se rellena con popularidad
reciente (`recent_days`, igual que el resto de modelos), de modo que la diferencia
frente a ellos se deba sólo a la parte colaborativa, que únicamente puede actuar
sobre usuarios "warm".

Uso: python -m src.models.collaborative
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from src.config import INTERACTIONS_PATH, RANDOM_STATE, REPORTS_DIR
from src.evaluation.evaluate import evaluate, temporal_split
from src.models.popularity import PopularityRecommender


class SVDRecommender:
    def __init__(self, n_factors=32, recent_days=30, rating_weight=False):
        # rating_weight=True: la matriz usa review_score/5 (faltantes = promedio) en vez de 1.
        self.n_factors = n_factors
        self.recent_days = recent_days
        self.rating_weight = rating_weight

    def fit(self, train):
        df = train[["customer_unique_id", "product_id", "review_score"]].copy()
        if self.rating_weight:
            df["w"] = df["review_score"].fillna(df["review_score"].mean()) / 5.0
        else:
            df["w"] = 1.0
        u_codes, self.users_ = pd.factorize(df["customer_unique_id"])
        i_codes, self.items_ = pd.factorize(df["product_id"])
        pairs = (pd.DataFrame({"u": u_codes, "i": i_codes, "w": df["w"].to_numpy()})
                 .groupby(["u", "i"], as_index=False)["w"].mean())
        self.R_ = sp.csr_matrix(
            (pairs["w"].to_numpy(), (pairs["u"].to_numpy(), pairs["i"].to_numpy())),
            shape=(len(self.users_), len(self.items_)))
        self.user_pos_ = {u: n for n, u in enumerate(self.users_)}

        n_comp = max(1, min(self.n_factors, min(self.R_.shape) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        self.U_ = svd.fit_transform(self.R_)   # usuarios x factores
        self.V_ = svd.components_              # factores x productos
        self.fallback_ = PopularityRecommender(recent_days=self.recent_days).fit(train)
        return self

    def _seen_idx(self, user_id):
        pos = self.user_pos_.get(user_id)
        if pos is None:
            return None, None
        s, e = self.R_.indptr[pos], self.R_.indptr[pos + 1]
        return pos, self.R_.indices[s:e]

    def recommend(self, user_id, k=10):
        pos, seen = self._seen_idx(user_id)
        recs = []
        if pos is not None and len(seen):
            scores = self.U_[pos] @ self.V_
            scores[seen] = -np.inf  # no recomendar lo ya comprado
            # orden estable: empates se resuelven por posición, resultado determinista
            top = np.argsort(-scores, kind="stable")[:k]
            recs = [self.items_[i] for i in top if np.isfinite(scores[i])]
        if len(recs) < k:
            taken = set(recs)
            recs += [p for p in self.fallback_.recommend(user_id, k + len(recs))
                     if p not in taken]
        return recs[:k]


def main():
    df = pd.read_parquet(INTERACTIONS_PATH)
    train, test = temporal_split(df)
    print(f"train: {len(train):,} interacciones | test: {len(test):,} interacciones")
    models = {f"svd_{f}": SVDRecommender(n_factors=f) for f in (8, 32, 64)}
    models["svd_32_rating"] = SVDRecommender(n_factors=32, rating_weight=True)
    results = pd.concat([evaluate(m.fit(train), train, test, name=n) for n, m in models.items()])
    pd.set_option("display.width", 160)
    print(results[results["k"] == 10].round(4).to_string(index=False))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "collaborative_results.csv"
    results.to_csv(out, index=False)
    print(f"Resultados guardados en {out}")


if __name__ == "__main__":
    main()

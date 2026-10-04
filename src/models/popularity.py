"""Baselines de popularidad.

- PopularityRecommender: top-N global por número de compradores distintos,
  opcionalmente ponderado por el review_score promedio (suavizado bayesiano).
- CategoryPopularityRecommender: para usuarios con historial, recomienda lo más
  popular de sus categorías favoritas; rellena con el top global (cold start).

Ambos excluyen productos que el usuario ya compró.

Uso:  python -m src.models.popularity
"""
import pandas as pd

from src.config import INTERACTIONS_PATH, REPORTS_DIR
from src.evaluation.evaluate import evaluate, temporal_split


class PopularityRecommender:
    def __init__(self, rating_weight=False, prior_weight=10):
        self.rating_weight = rating_weight
        self.prior_weight = prior_weight

    def _scores(self, train):
        g = train.groupby("product_id").agg(
            buyers=("customer_unique_id", "nunique"), rating=("review_score", "mean"))
        score = g["buyers"].astype(float)
        if self.rating_weight:
            mu = train["review_score"].mean()
            n = g["buyers"]
            bayes = (n * g["rating"].fillna(mu) + self.prior_weight * mu) / (n + self.prior_weight)
            score = score * bayes / 5
        return score.sort_values(ascending=False)

    def fit(self, train):
        self.ranking_ = self._scores(train)
        self.top_ = list(self.ranking_.index)
        self.history_ = train.groupby("customer_unique_id")["product_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id, k=10):
        seen = self.history_.get(user_id, set())
        return [p for p in self.top_[: k + len(seen)] if p not in seen][:k]


class CategoryPopularityRecommender(PopularityRecommender):
    def fit(self, train):
        super().fit(train)
        self.cat_of_ = train.drop_duplicates("product_id").set_index("product_id")["category"]
        ranked = self.ranking_.rename("score").to_frame().join(self.cat_of_)
        self.top_by_cat_ = {c: list(g.index) for c, g in ranked.groupby("category", sort=False)}
        self.user_cats_ = (train.groupby(["customer_unique_id", "category"]).size()
                           .sort_values(ascending=False).reset_index()
                           .groupby("customer_unique_id")["category"].agg(list).to_dict())
        return self

    def recommend(self, user_id, k=10):
        seen = self.history_.get(user_id, set())
        recs = []
        for cat in self.user_cats_.get(user_id, []):
            recs += [p for p in self.top_by_cat_.get(cat, [])[: k + len(seen)]
                     if p not in seen and p not in recs]
            if len(recs) >= k:
                return recs[:k]
        fill = [p for p in self.top_[: 2 * k + len(seen)] if p not in seen and p not in recs]
        return (recs + fill)[:k]


def main():
    df = pd.read_parquet(INTERACTIONS_PATH)
    train, test = temporal_split(df)
    print(f"train: {len(train):,} interacciones | test: {len(test):,} interacciones")
    models = {
        "pop_global": PopularityRecommender(),
        "pop_global_rating": PopularityRecommender(rating_weight=True),
        "pop_categoria": CategoryPopularityRecommender(rating_weight=True),
    }
    results = pd.concat([evaluate(m.fit(train), train, test, name=n) for n, m in models.items()])
    pd.set_option("display.width", 160)
    print(results.round(4).to_string(index=False))
    out = REPORTS_DIR / "baseline_results.csv"
    results.to_csv(out, index=False)
    print(f"Resultados guardados en {out}")


if __name__ == "__main__":
    main()

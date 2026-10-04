"""Protocolo de evaluación común para TODOS los modelos del equipo.

Split temporal (config.TRAIN_END / TEST_END):
  - train: compras anteriores a TRAIN_END
  - test : compras en [TRAIN_END, TEST_END)
  - relevantes de un usuario = productos que compró en test y NO había comprado en train

Cada modelo debe exponer:
  model.fit(train_df)
  model.recommend(user_id, k) -> lista de product_id (sin repetir)

Métricas: Precision@K, Recall@K, HitRate@K, NDCG@K y cobertura de catálogo.
Se reportan para todos los usuarios de test y para el segmento "warm"
(usuarios con historial en train), porque ~97% de los usuarios de test son cold start.
"""
import math

import numpy as np
import pandas as pd

from src.config import K_VALUES, TEST_END, TRAIN_END


def temporal_split(df, train_end=TRAIN_END, test_end=TEST_END):
    train = df[df["purchase_ts"] < train_end]
    test = df[(df["purchase_ts"] >= train_end) & (df["purchase_ts"] < test_end)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def ground_truth(train, test):
    """dict user -> set de productos nuevos comprados en test."""
    seen = train.groupby("customer_unique_id")["product_id"].agg(set)
    truth = {}
    for user, items in test.groupby("customer_unique_id")["product_id"]:
        new = set(items) - seen.get(user, set())
        if new:
            truth[user] = new
    return truth


def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant)


def hit_rate_at_k(recs, relevant, k):
    return float(bool(set(recs[:k]) & relevant))


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(recs[:k]) if item in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


METRICS = {"precision": precision_at_k, "recall": recall_at_k,
           "hit_rate": hit_rate_at_k, "ndcg": ndcg_at_k}


def evaluate(model, train, test, k_values=K_VALUES, name=None):
    truth = ground_truth(train, test)
    warm_users = set(train["customer_unique_id"])
    kmax = max(k_values)
    recs = {u: list(model.recommend(u, kmax)) for u in truth}
    n_catalog = train["product_id"].nunique()

    rows = []
    for segment, users in [("all", list(truth)), ("warm", [u for u in truth if u in warm_users])]:
        for k in k_values:
            row = {"model": name or type(model).__name__, "segment": segment, "k": k,
                   "n_users": len(users)}
            for m, fn in METRICS.items():
                row[f"{m}@k"] = np.mean([fn(recs[u], truth[u], k) for u in users]) if users else np.nan
            recommended = {i for u in users for i in recs[u][:k]}
            row["coverage"] = len(recommended) / n_catalog
            rows.append(row)
    return pd.DataFrame(rows)

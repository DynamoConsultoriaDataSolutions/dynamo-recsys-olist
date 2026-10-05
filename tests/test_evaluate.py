import pandas as pd
import pytest

from src.evaluation.evaluate import (ground_truth, hit_rate_at_k, ndcg_at_k,
                                     precision_at_k, recall_at_k, temporal_split)
from src.models.popularity import PopularityRecommender


def test_metricas_basicas():
    recs, rel = ["a", "b", "c", "d"], {"b", "z"}
    assert precision_at_k(recs, rel, 2) == 0.5
    assert recall_at_k(recs, rel, 2) == 0.5
    assert hit_rate_at_k(recs, rel, 1) == 0.0
    assert hit_rate_at_k(recs, rel, 2) == 1.0
    assert ndcg_at_k(["b", "z"], rel, 2) == pytest.approx(1.0)


@pytest.fixture
def toy():
    return pd.DataFrame({
        "customer_unique_id": ["u1", "u2", "u2", "u1", "u3"],
        "product_id": ["p1", "p1", "p2", "p3", "p1"],
        "purchase_ts": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-01",
                                       "2018-07-01", "2018-07-02"]),
        "review_score": [5, 4, 3, 5, 1],
        "category": ["x", "x", "y", "y", "x"],
    })


def test_split_y_ground_truth(toy):
    train, test = temporal_split(toy, "2018-06-01", "2018-09-01")
    assert len(train) == 3 and len(test) == 2
    truth = ground_truth(train, test)
    assert truth == {"u1": {"p3"}, "u3": {"p1"}}


def test_popularidad_excluye_vistos(toy):
    train, _ = temporal_split(toy, "2018-06-01", "2018-09-01")
    model = PopularityRecommender().fit(train)
    assert model.recommend("u_nuevo", 2) == ["p1", "p2"]
    assert "p1" not in model.recommend("u1", 2)

"""Pruebas del modelo SVD con un dataset mínimo hecho a mano."""
import pandas as pd
import pytest

from src.models.collaborative import SVDRecommender


@pytest.fixture
def train():
    rows = [
        # u1 y u2 compraron A y B; u3 sólo A -> a u3 hay que recomendarle B
        ("u1", "A", 5), ("u1", "B", 5),
        ("u2", "A", 4), ("u2", "B", 4),
        ("u3", "A", 5),
        # producto popular e independiente
        ("u4", "P", 5), ("u5", "P", 5), ("u6", "P", 4),
    ]
    df = pd.DataFrame(rows, columns=["customer_unique_id", "product_id", "review_score"])
    df["category"] = "x"
    df["purchase_ts"] = pd.date_range("2018-01-01", periods=len(df), freq="D")
    return df


def test_recommends_co_purchased_item(train):
    recs = SVDRecommender(n_factors=2, recent_days=None).fit(train).recommend("u3", 3)
    assert recs[0] == "B"
    assert "A" not in recs  # nunca lo ya comprado


def test_cold_start_falls_back_to_popularity(train):
    recs = SVDRecommender(n_factors=2, recent_days=None).fit(train).recommend("nuevo", 2)
    assert len(recs) == 2
    assert recs[0] in {"A", "P"}  # los más comprados


def test_no_duplicates_and_length(train):
    model = SVDRecommender(n_factors=2, recent_days=None).fit(train)
    for user in ["u1", "u3", "u6", "nuevo"]:
        recs = model.recommend(user, 3)
        assert len(recs) == len(set(recs)) <= 3


def test_deterministic(train):
    a = SVDRecommender(n_factors=2, recent_days=None).fit(train).recommend("u3", 3)
    b = SVDRecommender(n_factors=2, recent_days=None).fit(train).recommend("u3", 3)
    assert a == b


def test_rating_weight_runs(train):
    recs = SVDRecommender(n_factors=2, recent_days=None, rating_weight=True).fit(train).recommend("u3", 2)
    assert recs[0] == "B"

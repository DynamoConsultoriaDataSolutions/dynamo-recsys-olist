"""Pruebas del modelo exportable: guardar, cargar y recomendar con motivo."""
import pandas as pd
import pytest

from src.models.item_based import ItemToItemRecommender
from src.models.predict import example_users, is_known, load_bundle, recommend, user_history
from src.models.train_final import build_bundle, save_bundle


@pytest.fixture
def df():
    rows = [
        ("o1", "u1", "A", "x", 10, "2018-01-01"), ("o1", "u1", "B", "y", 20, "2018-01-01"),
        ("o2", "u2", "A", "x", 10, "2018-01-03"), ("o2", "u2", "B", "y", 20, "2018-01-03"),
        ("o3", "u3", "C", "x", 11, "2018-01-05"), ("o4", "u4", "C", "x", 11, "2018-01-06"),
        ("o5", "u5", "D", "x", 50, "2018-01-07"),
        ("o6", "u6", "E", "z", 5, "2018-05-30"), ("o7", "u7", "E", "z", 5, "2018-05-31"),
        ("o8", "u11", "A", "x", 10, "2018-02-01"),
    ]
    d = pd.DataFrame(rows, columns=["order_id", "customer_unique_id", "product_id", "category", "price", "purchase_ts"])
    d["purchase_ts"] = pd.to_datetime(d["purchase_ts"])
    d["review_score"] = 5.0
    return d


def test_reasons_match_recommend(df):
    model = ItemToItemRecommender(recent_days=30).fit(df)
    for user in ["u11", "u3", "nuevo"]:
        pairs = model.recommend_with_reasons(user, 4)
        assert [p for p, _ in pairs] == model.recommend(user, 4)
        assert {r for _, r in pairs} <= {"co-compra", "categoria", "tendencia"}
    assert model.recommend_with_reasons("u11", 2) == [("B", "co-compra"), ("C", "categoria")]
    assert all(r == "tendencia" for _, r in model.recommend_with_reasons("nuevo", 2))


def test_save_load_roundtrip(df, tmp_path):
    bundle = build_bundle(df)
    path = save_bundle(bundle, tmp_path / "modelo.joblib")
    loaded = load_bundle(path)
    assert loaded["model"].recommend("u11", 3) == bundle["model"].recommend("u11", 3)
    assert loaded["trained_until"] == "2018-05-31 00:00:00"


def test_recommend_table(df, tmp_path):
    bundle = load_bundle(save_bundle(build_bundle(df), tmp_path / "m.joblib"))
    out = recommend(bundle, "u11", 3)
    assert list(out.columns) == ["posicion", "product_id", "category", "price", "motivo", "explicacion"]
    assert out["posicion"].tolist() == [1, 2, 3]
    assert out["explicacion"].notna().all()
    assert "A" not in out["product_id"].tolist()          # nunca lo ya comprado
    assert is_known(bundle, "u11") and not is_known(bundle, "nuevo")
    assert user_history(bundle, "u1")["product_id"].tolist() == ["A", "B"]
    assert example_users(bundle, n=2) == ["u1", "u2"]


def test_hasta_excludes_future(df):
    bundle = build_bundle(df, hasta="2018-05-01")
    assert bundle["trained_until"] < "2018-05-01"
    assert "E" not in set(bundle["products"]["product_id"])


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_final"):
        load_bundle(tmp_path / "no_existe.joblib")

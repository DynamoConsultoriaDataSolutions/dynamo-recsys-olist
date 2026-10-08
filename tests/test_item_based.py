import pandas as pd
import pytest

from src.models.item_based import ItemToItemRecommender
from src.models.popularity import PopularityRecommender


@pytest.fixture
def train():
    # u1 y u2 compran A y B juntos -> A y B son vecinos por co-compra.
    # C y D son de la misma categoría que A (contenido). E es lo más popular reciente.
    rows = [
        ("u1", "A", "x", 10, "2018-01-01"), ("u1", "B", "y", 20, "2018-01-02"),
        ("u2", "A", "x", 10, "2018-01-03"), ("u2", "B", "y", 20, "2018-01-04"),
        ("u3", "C", "x", 11, "2018-01-05"), ("u4", "C", "x", 11, "2018-01-06"),
        ("u5", "D", "x", 50, "2018-01-07"),
        ("u6", "E", "z", 5, "2018-05-30"), ("u7", "E", "z", 5, "2018-05-31"),
        ("u8", "F", "z", 5, "2017-01-01"), ("u9", "F", "z", 5, "2017-01-02"),
        ("u10", "F", "z", 5, "2017-01-03"),
        ("u11", "A", "x", 10, "2018-02-01"),
    ]
    df = pd.DataFrame(rows, columns=["customer_unique_id", "product_id", "category", "price", "purchase_ts"])
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    df["review_score"] = 5.0
    return df


def test_popularidad_reciente_cambia_el_ranking(train):
    todo = PopularityRecommender().fit(train)
    reciente = PopularityRecommender(recent_days=30).fit(train)
    assert todo.recommend("nuevo", 1) == ["A"]          # A: 3 compradores en total
    assert reciente.recommend("nuevo", 1) == ["E"]      # E: lo único vendido en el último mes


def test_cocompra_primero_luego_contenido(train):
    model = ItemToItemRecommender(recent_days=30).fit(train)
    recs = model.recommend("u11", 3)                    # u11 compró A
    assert recs[0] == "B"                               # vecino por co-compra
    assert recs[1:] == ["C", "D"]                       # misma categoría, por popularidad
    assert "A" not in recs                              # nunca recomienda lo ya comprado


def test_usuario_nuevo_usa_popularidad_reciente(train):
    model = ItemToItemRecommender(recent_days=30).fit(train)
    assert model.recommend("nuevo", 1) == ["E"]


def test_similitud_coseno(train):
    model = ItemToItemRecommender().fit(train)
    sims = dict(model.neighbors_["A"])
    # A y B: 2 clientes en común; A la compraron 2 de los usuarios con 2+ productos, B también
    assert sims["B"] == pytest.approx(1.0)

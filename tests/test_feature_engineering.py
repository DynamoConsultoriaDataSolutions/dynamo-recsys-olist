import pandas as pd

from src.data.feature_engineering import (
    build_user_features,
    build_product_features,
)


def sample_data():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_unique_id": ["u1", "u1", "u2"],
        "product_id": ["p1", "p2", "p1"],
        "purchase_ts": pd.to_datetime([
            "2018-01-01",
            "2018-02-01",
            "2018-03-01",
        ]),
        "price": [100.0, 50.0, 100.0],
        "quantity": [2, 1, 1],
        "review_score": [5.0, None, 4.0],
        "category": ["electronics", "books", "electronics"],
    })


def test_user_features():
    df = sample_data()

    features = build_user_features(
        df,
        cutoff="2018-06-01"
    )

    assert features["customer_unique_id"].is_unique
    assert features["user_avg_review"].isna().sum() == 0
    assert (features["user_recency_days"] >= 0).all()

    user_1 = features[
        features["customer_unique_id"] == "u1"
    ].iloc[0]

    assert user_1["user_total_spent"] == 250.0


def test_product_features():
    df = sample_data()

    features = build_product_features(df)

    assert features["product_id"].is_unique
    assert features["prod_avg_review"].isna().sum() == 0
    assert (features["prod_total_quantity"] > 0).all()
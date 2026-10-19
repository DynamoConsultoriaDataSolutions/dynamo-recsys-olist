import pandas as pd
import pytest

from src.dashboard_data import (business_summary, filter_business, load_business,
                                monthly_sales, select_confidence, select_metrics)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/raw").mkdir()
    df = pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o3"],
        "customer_unique_id": ["u1", "u1", "u1", "u2"],
        "product_id": ["p1", "p2", "p1", "p3"],
        "purchase_ts": pd.to_datetime(["2018-01-01 23:59", "2018-01-01 23:59",
                                       "2018-02-01 00:00", "2018-02-02 00:00"]),
        "category": ["baby", "toys", "baby", "toys"],
        "price": [10., 20., 10., 30.], "quantity": [2, 1, 3, 1],
    })
    df.to_parquet(tmp_path / "data/processed/interactions.parquet")
    pd.DataFrame({"order_id": ["o1", "o2", "o3"], "customer_id": ["c1", "c2", "c3"]}).to_csv(
        tmp_path / "data/raw/olist_orders_dataset.csv", index=False)
    pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "RJ", "SP"]}).to_csv(
        tmp_path / "data/raw/olist_customers_dataset.csv", index=False)
    return tmp_path


def test_sales_use_quantities_and_distinct_orders(root):
    df = load_business(root)
    summary = business_summary(df)
    assert summary["sales_brl"] == 100
    assert summary["orders"] == 3
    assert summary["customers"] == 2
    assert summary["units"] == 7
    assert summary["repeat_customers_pct"] == 50
    assert monthly_sales(df)["sales_brl"].tolist() == [40, 60]


def test_filters_include_entire_end_day_and_use_order_geography(root):
    df = load_business(root)
    january = filter_business(df, "2018-01-01", "2018-01-01", categories=["baby"], states=["SP"])
    assert january["order_id"].tolist() == ["o1"]
    assert business_summary(january)["sales_brl"] == 20
    # El mismo cliente aparece en RJ en su segundo pedido; no se usa su primera dirección.
    rio = filter_business(df, "2018-01-01", "2018-02-02", states=["RJ"])
    assert rio["order_id"].tolist() == ["o2"]
    assert filter_business(df, "2018-01-01", "2018-02-02", categories=[]).empty


def test_empty_and_missing_geography_are_supported(root):
    (root / "data/raw/olist_orders_dataset.csv").unlink()
    df = load_business(root)
    assert df["customer_state"].eq("Sin dato").all()
    empty = filter_business(df, "2019-01-01", "2019-01-02")
    assert business_summary(empty)["sales_per_order_brl"] == 0
    assert business_summary(empty)["repeat_customers_pct"] == 0


def test_bad_dates_and_duplicate_geography_do_not_silently_inflate_sales(root):
    df = load_business(root)
    with pytest.raises(ValueError):
        filter_business(df, "2018-02-01", "2018-01-01")
    path = root / "data/raw/olist_customers_dataset.csv"
    customers = pd.read_csv(path)
    pd.concat([customers, customers.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(pd.errors.MergeError):
        load_business(root)


def test_metrics_do_not_mix_k_or_segments_and_ci_only_applies_to_k10():
    metrics = pd.DataFrame({"model": ["svd", "svd", "svd"], "segment": ["all", "warm", "warm"],
                            "k": [10, 10, 20], "hit_rate@k": [.01, .02, .03]})
    assert select_metrics(metrics, "warm", 10, ["svd"])["hit_rate@k"].tolist() == [.02]
    ci = pd.DataFrame({"comparacion": ["svd", "svd"], "segment": ["all", "warm"],
                       "estimado": [.01, .02]})
    assert select_confidence(ci, "warm", 20, ["svd"]).empty
    assert select_confidence(ci, "warm", 10, ["svd"])["estimado"].tolist() == [.02]

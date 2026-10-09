"""Entrena y compara todos los modelos con el mismo protocolo.

1. VALIDACIÓN (sin tocar el test): se entrena con compras < VAL_START y se valida
   en [VAL_START, TRAIN_END). Ahí se elige la ventana de popularidad reciente.
2. TEST: con la ventana elegida, se reentrena con todo < TRAIN_END y se evalúa
   en [TRAIN_END, TEST_END). Se guarda reports/model_comparison.csv.

Uso:  python -m src.models.train_compare
"""
import pandas as pd

from src.config import INTERACTIONS_PATH, REPORTS_DIR, TRAIN_END, VAL_START
from src.evaluation.evaluate import evaluate, temporal_split
from src.models.collaborative import SVDRecommender
from src.models.item_based import ItemToItemRecommender
from src.models.popularity import CategoryPopularityRecommender, PopularityRecommender

WINDOWS = [7, 14, 30, 60, 90, 120, 180]


def hit10(results, segment="all"):
    r = results[(results["k"] == 10) & (results["segment"] == segment)]
    return float(r["hit_rate@k"].iloc[0])


def select_window(df):
    train, val = temporal_split(df, VAL_START, TRAIN_END)
    rows = [{"recent_days": d,
             "hit_rate@10_val": hit10(evaluate(PopularityRecommender(recent_days=d).fit(train),
                                               train, val, k_values=(10,)))}
            for d in [None] + WINDOWS]
    table = pd.DataFrame(rows)
    best = int(table.dropna().sort_values("hit_rate@10_val", ascending=False)
               .iloc[0]["recent_days"])
    return best, table


def main():
    df = pd.read_parquet(INTERACTIONS_PATH)

    best, val_table = select_window(df)
    print("== 1. Validación: ventana de popularidad reciente ==")
    print(val_table.assign(**{"hit_rate@10_val": (val_table["hit_rate@10_val"] * 100).round(2)})
          .rename(columns={"hit_rate@10_val": "HitRate@10 val (%)"})
          .fillna({"recent_days": "todo"}).to_string(index=False))
    print(f"-> ventana elegida: {best} días\n")
    val_table.to_csv(REPORTS_DIR / "validation_recent_window.csv", index=False)

    train, test = temporal_split(df)
    models = {
        "pop_global (baseline)": PopularityRecommender(),
        "pop_categoria": CategoryPopularityRecommender(rating_weight=True),
        f"pop_reciente_{best}d": PopularityRecommender(recent_days=best),
        "item2item_solo_cocompra": ItemToItemRecommender(recent_days=best, use_content=False),
        "item2item_hibrido": ItemToItemRecommender(recent_days=best, use_content=True),
        "svd_colaborativo": SVDRecommender(n_factors=4, recent_days=best),
    }
    results = pd.concat([evaluate(m.fit(train), train, test, name=n) for n, m in models.items()])
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    k10 = results[results["k"] == 10].pivot(index="model", columns="segment",
                                           values=["hit_rate@k", "ndcg@k", "coverage"])
    k10 = k10.loc[list(models)]
    base = k10[("hit_rate@k", "all")].iloc[0]
    k10[("mejora_vs_baseline_%", "all")] = (k10[("hit_rate@k", "all")] / base - 1) * 100
    print("== 2. Test (K = 10) ==")
    pd.set_option("display.width", 180)
    print(k10.round(4).to_string())
    print(f"\nResultados completos en {REPORTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()

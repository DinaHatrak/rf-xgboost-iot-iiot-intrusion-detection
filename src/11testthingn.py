from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(
    r"C:\Users\LOQ\Desktop\Random Forest versus XGBoost"
) / "results" / "analysis"

for model in ["random_forest", "xgboost"]:

    print("\n" + "=" * 70)
    print(model.upper())
    print("=" * 70)

    dfs = {}

    for seed in [1, 2, 3]:
        f = BASE / f"10_edge_ablation_{model}_seed{seed}.csv"
        dfs[seed] = pd.read_csv(f)

    for a, b in [(1,2), (1,3), (2,3)]:

        pa = dfs[a]["prob_attack"].to_numpy()
        pb = dfs[b]["prob_attack"].to_numpy()

        ya = dfs[a]["y_pred"].to_numpy()
        yb = dfs[b]["y_pred"].to_numpy()

        d = np.abs(pa - pb)

        print(f"\nSeed {a} vs {b}")
        print("Different hard predictions:",
              np.sum(ya != yb))
        print("Probabilities exactly equal:",
              np.array_equal(pa, pb))
        print("Mean abs prob difference:",
              f"{d.mean():.10f}")
        print("Max abs prob difference:",
              f"{d.max():.10f}")
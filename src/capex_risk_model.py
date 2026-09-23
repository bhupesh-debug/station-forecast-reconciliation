"""CapEx overrun classifier. The decision threshold is chosen for high recall on
the overrun class (a missed overrun costs more than a false alarm)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import precision_score, recall_score, roc_auc_score

FEATURES = ["budget_musd", "planned_months", "vendor_count", "scope_changes",
            "is_regulatory", "prior_overruns_at_station"]
TARGET_RECALL = 0.85
SEED = 42


def pick_threshold(y, p, target=TARGET_RECALL):
    """Highest threshold that still reaches the target recall on validation data."""
    best = 0.0
    for t in np.linspace(0.05, 0.95, 91):
        if recall_score(y, p >= t) >= target:
            best = t
    return best


def main(path="data/synthetic/capex_projects.csv"):
    df = pd.read_csv(path)
    X, y = df[FEATURES], df["overrun"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                       random_state=SEED)
    oof = cross_val_predict(model, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
    thr = pick_threshold(ytr, oof)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = p >= thr
    res = {"n_projects": len(df), "overrun_base_rate": round(float(y.mean()), 3),
           "threshold": round(float(thr), 2), "target_recall": TARGET_RECALL,
           "test_recall": round(float(recall_score(yte, pred)), 3),
           "test_precision": round(float(precision_score(yte, pred)), 3),
           "test_auc": round(float(roc_auc_score(yte, p)), 3),
           "top_features": dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 3),
                                model.feature_importances_)), key=lambda kv: -kv[1])[:3])}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/capex_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

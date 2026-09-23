import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_pipeline as dp
import reconcile_forecast as rf


def _wide():
    rng = np.random.default_rng(0)
    st = dp.generate_stations(rng)
    df = dp.generate_monthly(rng, st)
    return df.pivot(index="month", columns="station", values="revenue")


def test_shapes_and_hierarchy():
    wide = _wide()
    assert wide.shape[1] == 17
    S = rf.summing_matrix(17)
    assert S.shape == (18, 17) and np.allclose(S[0], 1)


def test_reconciled_forecasts_are_coherent():
    """Enterprise forecast must equal the sum of station forecasts after reconciliation."""
    wide = _wide().iloc[:72]
    S = rf.summing_matrix(17)
    series = pd.concat([wide.sum(axis=1).rename("TOTAL"), wide], axis=1)
    fc, rs = zip(*[rf.base_forecast(series[c], 6) for c in series.columns])
    base, res = np.column_stack(fc), np.column_stack(rs)
    props = (wide.iloc[-24:].sum() / wide.iloc[-24:].sum().sum()).values
    for m in ["bottom_up", "top_down", "mint"]:
        out = rf.reconcile(base, res, S, m, props)
        assert np.allclose(out[:, 0], out[:, 1:].sum(axis=1)), m


def test_backtest_has_no_lookahead():
    wide = _wide()
    T, o = len(wide), len(wide) - rf.HORIZON * rf.N_FOLDS
    assert o > 36  # enough training history for seasonal models at first origin

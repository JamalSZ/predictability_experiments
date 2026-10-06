"""Pipeline / plumbing tests: dataset loader, benchmark runner, experiment
runners on tiny synthetic inputs.  The CSV written to tmp_path is synthetic
Markov data used only to exercise the loader -- it is not a stand-in for the
real datasets."""
import argparse
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from predictability.benchmarks import eps_accuracy, eps_accuracy_sweep, run_fomm, run_statsforecast, snap_to_values, train_test_split_index
from predictability.datasets import MAX_N_HARD_LIMIT, load_all, load_config
from predictability.synthetic import generate

warnings.filterwarnings("ignore")


@pytest.fixture
def synthetic_config(tmp_path):
    _, x, _ = generate("N5Ds", 400, seed=0)
    csv = tmp_path / "series.csv"
    pd.DataFrame({"date": np.arange(400), "value": x, "other": 1.0}).to_csv(csv, index=False)
    cfg = {"defaults": {"max_n": 300, "slice": "head"}, "datasets": [{"name": "Synth", "path": str(csv), "column": "value", "season_length": 1}]}
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, x


def test_split_index():
    assert train_test_split_index(100) == 80
    assert train_test_split_index(5, 0.2) == 4


def test_loader_truncates_and_reads_column(synthetic_config):
    path, x = synthetic_config
    specs = load_config(str(path))
    assert specs[0].max_n == 300 and specs[0].column == "value"
    ds = load_all(str(path))[0]
    assert ds.n == 300 and np.allclose(ds.values, x[:300])
    assert ds.season_length == 1


def test_loader_hard_cap(tmp_path):
    csv = tmp_path / "big.csv"
    pd.DataFrame({"v": np.arange(6000, dtype=float)}).to_csv(csv, index=False)
    cfg = {"datasets": [{"name": "Big", "path": str(csv), "column": "v", "max_n": 99999, "slice": "tail"}]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    ds = load_all(str(p))[0]
    assert ds.n == MAX_N_HARD_LIMIT and ds.values[-1] == 5999


def test_loader_missing_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"datasets": [{"name": "X", "path": "/does/not/exist.csv", "column": "a"}]}))
    with pytest.raises(FileNotFoundError):
        load_all(str(p))


def test_eps_accuracy_definitions():
    df = pd.DataFrame({"dataset": "d", "model": "m", "t": [0, 1, 2, 3], "true": [0.0, 1.0, 2.0, 3.0], "predicted": [0.0, 1.5, 2.5, 4.0]})
    assert eps_accuracy(df, 0.5)["accuracy"].iloc[0] == 0.75  # <= eps
    assert eps_accuracy(df, 0.5, strict=True)["accuracy"].iloc[0] == 0.25  # < eps
    sweep = eps_accuracy_sweep(df, [0.0, 1.0])
    assert sweep["accuracy"].tolist() == [0.25, 1.0]


def test_statsforecast_runner_output_format():
    _, x, _ = generate("N5Ds", 200, seed=2)
    preds = run_statsforecast(x, "S", models=["Naive", "WindowAverage", "AutoTheta"], season_length=1)
    assert list(preds.columns) == ["dataset", "model", "t", "true", "predicted"]
    assert set(preds.model) == {"Naive", "WindowAverage", "AutoTheta"}
    split = train_test_split_index(200)
    naive = preds[preds.model == "Naive"].sort_values("t")
    assert naive.t.tolist() == list(range(split, 200))
    assert np.allclose(naive["true"], x[split:])
    assert np.allclose(naive["predicted"], x[split - 1 : -1])  # naive = previous value
    acc = eps_accuracy(preds, 0.0)
    assert len(acc) == 3 and (acc.n_pred == 200 - split).all()


def test_seasonal_naive_skipped_without_season():
    _, x, _ = generate("N5Du", 100, seed=0)
    preds = run_statsforecast(x, "S", models=["SeasonalNaive"], season_length=1)
    assert preds.empty
    preds = run_statsforecast(x, "S", models=["SeasonalNaive"], season_length=4)
    assert set(preds.model) == {"SeasonalNaive"}


def test_fomm_beats_naive_on_skewed_chain_and_snap():
    chain, x, _ = generate("N5Ds", 600, seed=5)
    fomm = run_fomm(x, "S")
    naive = run_statsforecast(x, "S", models=["Naive"])
    assert eps_accuracy(fomm, 0.0)["accuracy"].iloc[0] > eps_accuracy(naive, 0.0)["accuracy"].iloc[0]
    snapped = snap_to_values(naive.assign(predicted=naive.predicted + 1e-6), chain.values)
    assert set(np.round(snapped.predicted, 12)) <= set(np.round(chain.values, 12))


def _ns(mod, **overrides):
    p = argparse.ArgumentParser()
    mod.add_args(p)
    ns = p.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_exp1_exp2_exp3_small(tmp_path):
    from predictability.experiments import exp1_upper_bound, exp2_entropy_accuracy, exp3_runtime

    df1 = exp1_upper_bound.run(_ns(exp1_upper_bound, out=str(tmp_path / "e1"), n=150, states="3,5", dists="u,s", benchmarks="Naive"))
    assert len(df1) == 4 and (tmp_path / "e1" / "exp1_Du.png").exists()
    assert np.allclose(df1.pi_max_ifi, df1.pi_max_rnlj)
    df2 = exp2_entropy_accuracy.run(_ns(exp2_entropy_accuracy, out=str(tmp_path / "e2"), n=150, skew_indices="0,7", uniform_states="2,5"))
    assert df2.ifi_equals_rnlj.all() and len(df2) == 8
    df3 = exp3_runtime.run(_ns(exp3_runtime, out=str(tmp_path / "e3"), sizes="60,120", repeats=1, eps_levels="0,0.5,1"))
    assert set(df3.algorithm) == {"bf", "rnlj", "iesj", "ifi"} and df3.time_s_median.gt(0).all()
    assert (tmp_path / "e3" / "exp3_results.csv").exists()


def test_exp4_exp5_with_local_config(synthetic_config, tmp_path):
    from predictability.experiments import exp4_eps_sensitivity, exp5_model_independence

    path, _ = synthetic_config
    df4 = exp4_eps_sensitivity.run(_ns(exp4_eps_sensitivity, config=str(path), out=str(tmp_path / "e4"), eps="0,0.05,0.2", models="Naive,AutoTheta"))
    assert len(df4) == 3 and {"pi_Naive", "pi_AutoTheta", "pi_max", "N_range", "N_recurrence"} <= set(df4.columns)
    assert df4.sort_values("eps").pi_max.is_monotonic_increasing  # more tolerance -> higher bound
    assert (tmp_path / "e4" / "exp4_Synth.png").exists()
    df5 = exp5_model_independence.run(_ns(exp5_model_independence, config=str(path), synthetic="N5Du", n=150, out=str(tmp_path / "e5"), models="Naive"))
    assert len(df5) == 2 and {"pi_max_discretized", "pi_FOMM"} <= set(df5.columns)
    assert (tmp_path / "e5" / "exp5_spearman.csv").exists()


def test_cli_estimate(synthetic_config, tmp_path, capsys):
    from predictability.cli import main

    path, _ = synthetic_config
    assert main(["-q", "estimate", "--config", str(path), "--eps", "0,0.1", "--out", str(tmp_path / "est")]) == 0
    out = pd.read_csv(tmp_path / "est" / "estimates.csv")
    assert len(out) == 2 and out.H_est.iloc[0] >= out.H_est.iloc[1]

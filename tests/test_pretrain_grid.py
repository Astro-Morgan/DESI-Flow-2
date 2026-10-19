"""
Smoke test of the grid Eidos pretraining entry point (src/DESIFlow/training/pretrain_grid.py) on a tiny model and a few steps (CPU): a run writes its outputs, the gate/ramp/heads machinery
runs, and --resume continues from last.pt. Run directly (python tests/test_pretrain_grid.py) or with pytest.
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.training import pretrain_grid
from DESIFlow.training.masking import N_PIX

TMP = Path(tempfile.mkdtemp(prefix="pretrain_grid_test_"))
DATA = TMP / "data"


def make_dataset(path, n=40):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    wave = np.linspace(3600, 9824, N_PIX)
    z = rng.uniform(0.1, 1.5, n)
    flux = np.stack([5 * (wave / 6000.) ** rng.uniform(-2, 1) * (1 + 0.3 * np.exp(-0.5 * ((wave - 3727 * (1 + zi)) / 6) ** 2)) + rng.normal(0, 1, N_PIX) for zi in z]).astype(np.float32)
    np.save(path / "flux.npy", flux)
    np.save(path / "ivar.npy", np.ones((n, N_PIX), np.float32))
    np.save(path / "mask.npy", np.zeros((n, N_PIX), np.int32))
    np.save(path / "z.npy", z)


make_dataset(DATA)
COMMON = ["--data-dir", str(DATA), "--model", "small", "--dim", "32", "--layers", "1", "--hyb-dim", "32", "--batch", "4", "--accum", "2", "--val-n", "6", "--test-n", "6", "--eval-batch", "3",
          "--eval-every", "3", "--recon-every", "3", "--save-every", "3", "--log-every", "1", "--warmup", "2", "--gate-min-step", "0", "--gate-thr", "10", "--gate-ramp-steps", "2",
          "--ramp-abs", "1", "3", "--cpu", "--workers", "0", "--plot-z", "0.5,1.0"]


def test_run_resume_and_outputs():
    out = TMP / "run"
    s1 = pretrain_grid.main(COMMON + ["--out", str(out), "--steps", "6"])
    assert s1["steps_done"] == 6 and (out / "last.pt").exists() and (out / "best.pt").exists() and (out / "eval_log.jsonl").exists()
    rows = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert len(rows) == 6 and all(np.isfinite(r["loss_rec"]) for r in rows)
    assert rows[0]["zscale"] == 0.0 and rows[-1]["zscale"] == 1.0, "the z-gradient ramp must go 0 -> 1"
    assert rows[0]["gate"] == 0.0 and rows[-1]["gate"] == 1.0 and s1["gate_trigger"] is not None, "the precision gate must open (threshold set to 10) and ramp to 1"
    ev = [json.loads(l) for l in open(out / "eval_log.jsonl")]
    assert len(ev) == 2 and np.isfinite(ev[-1]["val_wmse"]) and "z_sigma_nmad" in ev[-1]
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["step"] == 6 and ck["trig"] == s1["gate_trigger"] and ck["heads"] is not None
    s2 = pretrain_grid.main(COMMON + ["--out", str(out), "--steps", "8", "--resume"])
    rows2 = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert s2["steps_done"] == 8 and rows2[-1]["step"] == 8 and rows2[-3]["step"] == 6 and s2["gate_trigger"] == s1["gate_trigger"]
    return f"6 steps (accum 2, heads, ramp 0->1, gate opened at step {s1['gate_trigger']}), outputs written, resumed 6 -> 8 with the gate state kept; val_wmse {ev[-1]['val_wmse']:.3f}"


def test_no_heads_run():
    out = TMP / "run_nohead"
    s = pretrain_grid.main(COMMON + ["--out", str(out), "--steps", "3", "--no-heads"])
    rows = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert s["steps_done"] == 3 and "z_loss" not in rows[0] and np.isfinite(rows[-1]["loss_rec"])
    return "reconstruction-only run (no heads, no z feed) works"


def test_default_position_scheme_and_other_schemes():
    out = TMP / "run_pos"
    s = pretrain_grid.main(COMMON + ["--out", str(out), "--steps", "2", "--no-heads"])
    cfg = json.load(open(out / "config.json"))["args"]
    assert cfg["pos"] == "cbias" and s["steps_done"] == 2, "cbias must be the default position scheme"
    ck = torch.load(out / "last.pt", weights_only=False)
    assert any(k.endswith(".R") for k in ck["model"]) and any(k.endswith(".g.2.weight") for k in ck["model"])
    for pos in ("bias", "rope"):
        s2 = pretrain_grid.main(COMMON + ["--out", str(TMP / f"run_{pos}"), "--steps", "2", "--no-heads", "--pos", pos])
        assert s2["steps_done"] == 2
    return "default --pos cbias (checkpoint holds g and R); --pos bias and --pos rope still train"


def test_real_data_safeguards_gate_force_and_auto_class_range():
    """A set whose redshifts reach z = 5.9 gets a wider classifier (auto: ln 6.9 = 1.93 -> 2.0, 1001 bins), a normal set keeps the benchmark's 1.6 (801 bins); a gate that can never open by precision
    (--gate-thr 0) still opens at --gate-force-step."""
    d2 = TMP / "data_highz"
    make_dataset(d2)
    z = np.load(d2 / "z.npy"); z[:3] = [5.9, 4.5, 0.2]; np.save(d2 / "z.npy", z)
    out = TMP / "run_highz"
    s = pretrain_grid.main(["--data-dir", str(d2)] + COMMON[2:] + ["--out", str(out), "--steps", "4", "--gate-thr", "0", "--gate-force-step", "2"])
    ck = torch.load(out / "last.pt", weights_only=False)
    n_bins = ck["heads"]["heads.cls.trunk.weight" if "heads.cls.trunk.weight" in ck["heads"] else "cls.trunk.weight"].shape[0] if ck["heads"] is not None else None
    assert n_bins == 1001, n_bins
    assert s["gate_trigger"] == 2, s["gate_trigger"]
    rows = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert rows[0]["gate"] == 0.0 and rows[-1]["gate"] > 0.0, [r["gate"] for r in rows]
    out2 = TMP / "run_lowz"
    pretrain_grid.main(COMMON + ["--out", str(out2), "--steps", "1", "--gate-thr", "0"])
    ck2 = torch.load(out2 / "last.pt", weights_only=False)
    n2 = [v.shape[0] for k, v in ck2["heads"].items() if k.endswith("cls.trunk.weight")][0]
    assert n2 == 801, n2
    return "z up to 5.9 -> 1001 classifier bins (benchmark 801 for z <= 1.5); a gate with threshold 0 opened at --gate-force-step 2"


def test_lsf_augmented_run():
    out = TMP / "run_lsf"
    s = pretrain_grid.main(COMMON + ["--out", str(out), "--steps", "3", "--no-heads", "--lsf-aug", "0.6", "--lsf-prob", "1.0"])
    rows = [json.loads(l) for l in open(out / "train_log.jsonl")]
    assert s["steps_done"] == 3 and np.isfinite(rows[-1]["loss_rec"])
    cfg = json.load(open(out / "config.json"))["args"]
    assert cfg["lsf_aug"] == 0.6 and cfg["lsf_prob"] == 1.0
    return "a run with --lsf-aug 0.6 --lsf-prob 1.0 (every object re-observed with a wider LSF) trains and logs finite losses"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")

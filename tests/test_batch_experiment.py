import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.batch_experiment import DIFFUTRON_CONFIGS, DSFLOW_CONFIGS


def test_diffutron_ddim_steps():
    ddim = next(c for c in DIFFUTRON_CONFIGS if c["label"] == "diffutron-ddim")
    assert ddim["steps"] == 50
    assert ddim["eta"] == 0.0


def test_diffutron_ddpm_steps():
    ddpm = next(c for c in DIFFUTRON_CONFIGS if c["label"] == "diffutron-ddpm")
    assert ddpm["steps"] == 1000
    assert ddpm["eta"] == 1.0


def test_dsflow_dpmpp_steps():
    dpm = next(c for c in DSFLOW_CONFIGS if c["label"] == "dsflow-dpm++")
    assert dpm["steps"] == 10
    assert dpm["solver"] == "dpm++"


def test_dsflow_euler_steps():
    euler = next(c for c in DSFLOW_CONFIGS if c["label"] == "dsflow-euler")
    assert euler["steps"] == 20
    assert euler["solver"] == "euler"


import tempfile
from scripts.batch_experiment import _needs_training


def test_needs_training_missing_artifact():
    with tempfile.TemporaryDirectory() as d:
        assert _needs_training(d, ["model.pt"]) is True


def test_needs_training_all_present():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model.pt"), "w").close()
        open(os.path.join(d, "tokenizer.pt"), "w").close()
        assert _needs_training(d, ["model.pt", "tokenizer.pt"]) is False


def test_needs_training_partial():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "model.pt"), "w").close()
        # tokenizer.pt is absent
        assert _needs_training(d, ["model.pt", "tokenizer.pt"]) is True

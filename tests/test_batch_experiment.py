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


import argparse


def _parse(args_list):
    """Helper: parse args the same way __main__ does."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--models', nargs='+', default=['diffutron', 'dsflow'],
                        choices=['diffutron', 'dsflow'])
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(args_list)


def test_default_models_both():
    args = _parse([])
    assert set(args.models) == {'diffutron', 'dsflow'}


def test_models_single():
    args = _parse(['--models', 'dsflow'])
    assert args.models == ['dsflow']


def test_datasets_default_none():
    args = _parse([])
    assert args.datasets is None


def test_datasets_explicit():
    args = _parse(['--datasets', 'adult', 'shoppers'])
    assert args.datasets == ['adult', 'shoppers']


def test_overwrite_default_false():
    args = _parse([])
    assert args.overwrite is False


def test_overwrite_flag():
    args = _parse(['--overwrite'])
    assert args.overwrite is True

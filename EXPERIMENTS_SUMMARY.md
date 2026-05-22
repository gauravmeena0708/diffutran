# DiffuTran / DiffuFlow Experiment Summary

This note summarizes the DiffuTran-family experiments run from the parent
`base_test2` benchmark harness. Results are stored outside this subrepo under
`/data/syntab/Gaurav/base_test2/workdir/runs/`.

## Methods and Variants

The active benchmark profile is `diffutran_four_variant` in the parent repo:

| Method | Family | Key settings |
|---|---|---|
| `diffutran-ddim` | DiffuTran | 50 inference steps, `eta=0.0` |
| `diffutran-ddpm` | DiffuTran | 1000 inference steps, `eta=1.0` |
| `diffuflow-dpm++` | DiffuFlow | 10 flow steps, `solver=dpm++` |
| `diffuflow-euler` | DiffuFlow | 20 flow steps, `solver=euler` |
| `guided-v1` | Baseline | Guided diffusion baseline from parent framework |
| `tabdiff-none` | Baseline | TabDiff baseline with no privacy/noise variant |

## Completed Runs

| Run | Dataset | Samples | Methods | Runs | Status | Main artifacts |
|---|---:|---:|---:|---:|---|---|
| `diffutran_four_variant_pilot` | `adult` | 100 | 4 | 16 | complete, 0 failed | `benchmark_results.csv`, `benchmark_summary.xlsx` |
| `diffutran_six_method_pilot` | `adult` | 10000 | 6 | 24 | complete, 0 failed | `benchmark_results.csv`, `benchmark_summary.xlsx`, Plotly HTMLs, research tables |

The six-method pilot adds `tabdiff-none` and `guided-v1` as baselines. It used
four adult constraints, producing `6 methods x 4 constraints = 24` completed
runs.

## Four-Variant Pilot Results

Mean values over the 16 `adult` pilot rows:

| Method | Composite | CSR | CondFid | LowDensity | PairAssoc | Wasserstein | C2ST | TSTR | SampleSec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `diffutran-ddpm` | 0.5872 | 1.0000 | 0.8737 | 0.8317 | 0.9069 | 0.7331 | 0.1035 | 0.8873 | 6.25 |
| `diffuflow-euler` | 0.5769 | 1.0000 | 0.8608 | 0.8771 | 0.9077 | 0.7089 | 0.1019 | 0.9194 | 4.25 |
| `diffuflow-dpm++` | 0.5763 | 1.0000 | 0.8646 | 0.8769 | 0.9085 | 0.7214 | 0.1064 | 0.9305 | 4.75 |
| `diffutran-ddim` | 0.5533 | 1.0000 | 0.8347 | 0.8154 | 0.8994 | 0.6563 | 0.0710 | 0.7938 | 4.25 |

Takeaway: within the small 100-sample pilot, `diffutran-ddpm` had the best
composite score, while both DiffuFlow variants were close and faster.

## Six-Method Pilot Results

Mean values over the 24 `adult` rows:

| Method | Composite | CSR | CondFid | LowDensity | PairAssoc | Wasserstein | C2ST | TSTR | SampleSec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tabdiff-none` | 0.7407 | 1.0000 | 0.9791 | 0.9771 | 0.9748 | 0.9524 | 0.2502 | 0.9955 | 5.75 |
| `guided-v1` | 0.6797 | 0.9994 | 0.9004 | 0.9166 | 0.9578 | 0.7867 | 0.2503 | 0.9632 | 6.00 |
| `diffutran-ddpm` | 0.6657 | 1.0000 | 0.8925 | 0.8470 | 0.9401 | 0.7691 | 0.2177 | 0.9249 | 60.00 |
| `diffuflow-euler` | 0.6552 | 1.0000 | 0.8796 | 0.8909 | 0.9381 | 0.7442 | 0.2289 | 0.9424 | 6.50 |
| `diffuflow-dpm++` | 0.6468 | 1.0000 | 0.8785 | 0.8882 | 0.9387 | 0.7450 | 0.2306 | 0.9347 | 6.50 |
| `diffutran-ddim` | 0.6182 | 1.0000 | 0.8516 | 0.8265 | 0.9380 | 0.6868 | 0.2037 | 0.9134 | 7.25 |

Takeaways:

- `tabdiff-none` was the strongest overall baseline in this pilot.
- `guided-v1` ranked second and remained close to `tabdiff-none` on CSR and
  utility.
- Among DiffuTran/DiffuFlow variants, `diffutran-ddpm` had the best composite
  score, but it was much slower at 10000 samples.
- `diffuflow-euler` was the best speed/quality tradeoff among the new flow and
  transformer variants: close to `diffutran-ddpm` but around 6.5s sampling time.
- `diffutran-ddim` was fastest among DiffuTran variants but had the weakest
  composite score in the six-method pilot.

## Generated Reports to Revisit

For the six-method pilot:

- Results CSV: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/benchmark_results.csv`
- Excel report: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/benchmark_summary.xlsx`
- Radar chart: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/plots/radar_method_profiles.html`
- Dynamic 3D Plotly chart: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/plots/interactive_all_metrics.html`
- Research tables: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/research_tables/`
- Comparison HTML: `/data/syntab/Gaurav/base_test2/workdir/runs/diffutran_six_method_pilot/guided_v1_vs_harpoon_tabdiff.html`

Note: `guided_v1_vs_harpoon_tabdiff.html` is only a filename. This run does not
include Harpoon; the content compares the six methods listed above.

## Rebuild Command

From `/data/syntab/Gaurav/base_test2`, regenerate the standard report bundle:

```bash
RUN=workdir/runs/diffutran_six_method_pilot
python scripts/report_cli.py run-bundle --run-dir "$RUN" \
  --methods tabdiff-none guided-v1 diffutran-ddpm diffuflow-euler diffuflow-dpm++ diffutran-ddim
```

Add significance testing if needed:

```bash
python scripts/report_cli.py run-bundle --run-dir "$RUN" \
  --methods tabdiff-none guided-v1 diffutran-ddpm diffuflow-euler diffuflow-dpm++ diffutran-ddim \
  --significance --significance-metrics composite csr alpha wasserstein tstr
```

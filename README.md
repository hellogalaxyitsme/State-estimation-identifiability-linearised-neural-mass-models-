# State estimation and identifiability in linearised neural mass models from scalp EEG

Code and results for the analysis of how much of each neural-mass state can be
recovered from scalp EEG: exact theory checks, simulations of three neural-mass
families, a template-head network, known intracerebral current sources and
simultaneous intracranial and scalp recordings. Section, figure and table
numbers refer to the article and its electronic supplementary material (ESM).

## Repository layout

```
neural_observability/          analysis library
    theory/                    filtering, smoothing, separation theorem, error floor,
                               neural-mass families, symmetries, atlas, nonlinear filters
    head_models/               template and individual finite-element head models
    known_sources/             Localize-MI measurements
    stimulation/               single-pulse stimulation cohort
    working_memory/            working-memory cohort
    statistics.py              subject-level tests and bootstrap intervals
scripts/
    simulations/               theory and simulation analyses
    known_sources/             Localize-MI analyses
    stimulation/               stimulation-cohort analyses
    working_memory/            working-memory analyses
    fetch_fsaverage.py         downloads the MNE fsaverage template head
configs/                       settings of the empirical analyses
results/                       stored results of every analysis
tests/                         numerical and symbolic checks
```

## Installation

The analyses were run with Python 3.10 on Linux (32 CPU cores; an NVIDIA RTX
A5000 GPU for the nonlinear filtering study).

```bash
pip install -r requirements.txt
pip install -e .
python scripts/fetch_fsaverage.py      # fsaverage template head into data/subjects
```

The individual head models of the stimulation cohort additionally require
SimNIBS 4.6 (CHARM segmentation and its FEM solver), installed separately; its
location is set by `software.simnibs_root` in `configs/stimulation_fem.json`.

## Tests

```bash
python -m pytest
```

`tests/test_theory.py` checks each theorem numerically: separation against the
full network filter, the right-half-plane bound and the imaginary-axis example
of theorem 2, the error floor against the exact mismatched filter, smoothing
against Gaussian conditioning, the minimal Wendling–Chauvel realisation, the
slow–fast exchange and the reconstruction identity of theorem 4.
`tests/test_symmetry_generators.py` verifies in exact symbolic arithmetic that
every continuous symmetry generator of ESM table S2 annihilates the coefficient
Jacobian for all parameter values. The remaining tests cover the library and the
empirical analyses; the finite-element tests need Linux.

## Analyses

| Analysis | Article | Scripts | Results |
|---|---|---|---|
| Operating-point atlas: recoverability, second input, error floors, finite-record forecast | §4.1–4.3, 4.5; figures 2, 3, 5a–d; ESM S7 | `simulations/run_atlas.py`, `simulations/summarize_atlas.py` | `results/simulations/atlas/` |
| Symmetries, generic rank, realisability, equivalence search | §4.5; ESM S4, S5 | `simulations/run_symmetry.py` | `results/simulations/symmetry/` |
| Fitting simulated recordings and limiting filter error | §4.5; figure 5e; ESM S10 | `simulations/run_finite_record_fits.py`, `simulations/run_finite_record_filter.py` | `results/simulations/finite_record_fits/` |
| Nonlinear filtering (particle and extended Kalman filters) | §4.4; figure 4; ESM S9 | `simulations/run_nonlinear_filtering.py`, `simulations/summarize_nonlinear_filtering.py` | `results/simulations/nonlinear_filtering/` |
| Template-head network and montage ceilings | §4.6; figure 6 | `simulations/run_network.py` | `results/simulations/network/` |
| Known intracerebral current sources (Localize-MI) | §4.7; figure 7; ESM S8 | `known_sources/run_localize_mi.py`, `known_sources/run_localize_mi_sensitivity.py`, `known_sources/run_localize_mi_orientation.py` | `results/known_sources/` |
| Stimulation cohort, spherical head models | §4.8; figure 8a; ESM S6 | `stimulation/run_spherical_models.py` | `results/stimulation/spherical/` |
| Stimulation cohort, individual finite-element head models and projected trajectories | §4.8; figure 8a; ESM S6 | `stimulation/run_segmentation.py`, `stimulation/run_head_model.py`, `stimulation/run_lead_fields.py`, `stimulation/run_fem_analysis.py` | `results/stimulation/fem/` |
| Stimulation cohort, matched wrong sources | §4.8; figure 8a; ESM S6 | `stimulation/run_matched_sources.py` | `results/stimulation/matched_sources/` |
| Working-memory cohort, data and template forward models | §4.8; ESM S6 | `working_memory/download_data.py`, `working_memory/prepare_data.py` | `results/working_memory/data/` |
| Working-memory cohort, generative state-space mapping | §4.8; figure 8b; ESM S6 | `working_memory/run_generative_model.py` | `results/working_memory/generative_model/` |
| Working-memory cohort, comparison of mappings | §4.8; figure 8b; ESM S6 | `working_memory/run_benchmark.py` | `results/working_memory/benchmark/` |

Script paths are relative to `scripts/`.

## Results

* `results/simulations/atlas/atlas_summary.json`: medians, quartiles, inclusion
  record, floors and forecasts; one record per operating point in `*_records.jsonl`.
* `results/simulations/symmetry/symmetry_results.json`: exact ranks, generators,
  realisability and equivalence-search solutions.
* `results/simulations/finite_record_fits/verification_fit.json`: every fitted
  recording; `verification_filter.json`: limiting filter errors and forecasts.
* `results/simulations/nonlinear_filtering/`: every filtering setting
  (`pf_results.json`), convergence runs and summaries.
* `results/simulations/network/network_results.json`.
* `results/known_sources/localize_mi/localize_mi_results.json`: session-level
  values and endpoints for three skull conductivities;
  `results/known_sources/sensitivity/localize_mi_sensitivity.json`: every
  measurement variant and the reference interval.
* `results/stimulation/` and `results/working_memory/`: `results.json` of each
  analysis, with subject-, session-, run- and candidate-level tables as CSV files.
  The finite-element folder also holds the eligibility tables, the automated
  head-model QC of every patient and the lead-field validation.

Random seeds are fixed in each script or configuration.

## Data

The datasets are public and are not redistributed here:

* Localize-MI: doi:[10.12751/g-node.1cc1ae](https://doi.org/10.12751/g-node.1cc1ae)
* Simultaneous stereo-EEG and high-density EEG during single-pulse stimulation
  (ccepcoreg): OSF, doi:[10.17605/OSF.IO/WSGZP](https://doi.org/10.17605/OSF.IO/WSGZP)
* Simultaneous scalp and intracranial EEG during working memory: OpenNeuro
  ds004752 v1.0.1, doi:[10.18112/openneuro.ds004752.v1.0.1](https://doi.org/10.18112/openneuro.ds004752.v1.0.1)

By default the configurations expect the stimulation dataset in
`datasets/ccepcoreg` and the working-memory dataset in `datasets/ds004752`
(created by `scripts/working_memory/download_data.py`); the Localize-MI scripts
take the dataset root with `--data`. The stored results contain derived
quantities only.

## Reproducing the analyses

Theory and simulations:

```bash
python scripts/simulations/run_atlas.py --workers 30        # about 30 min on 32 cores
python scripts/simulations/summarize_atlas.py
python scripts/simulations/run_symmetry.py                  # about 30 min
python scripts/simulations/run_finite_record_fits.py --workers 16
python scripts/simulations/run_finite_record_filter.py
python scripts/simulations/run_nonlinear_filtering.py --mode main      # about 7 h on one GPU
python scripts/simulations/run_nonlinear_filtering.py --mode convergence --seconds 10 \
    --regimes jansen_rit_focus,wendling_2002,next_generation_gamma
python scripts/simulations/summarize_nonlinear_filtering.py
python scripts/simulations/run_network.py --subjects-dir data/subjects
```

Known current sources:

```bash
python scripts/known_sources/run_localize_mi.py --data <Localize-MI root>
python scripts/known_sources/run_localize_mi_sensitivity.py --data <Localize-MI root>
python scripts/known_sources/run_localize_mi_orientation.py --data <Localize-MI root>
```

Stimulation cohort:

```bash
python scripts/stimulation/run_spherical_models.py
python scripts/stimulation/run_segmentation.py        # CHARM segmentation and automated QC
# visual review of results/stimulation/fem/headmodel_qc, decisions in visual_qc.csv
python scripts/stimulation/run_segmentation.py --post-charm-eligibility
python scripts/stimulation/run_lead_fields.py
python scripts/stimulation/run_fem_analysis.py
python scripts/stimulation/run_matched_sources.py
```

Working-memory cohort:

```bash
python scripts/working_memory/download_data.py
python scripts/working_memory/prepare_data.py
python scripts/working_memory/run_generative_model.py
python scripts/working_memory/run_benchmark.py
```

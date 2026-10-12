# PSNG - Polysomnography signal clustering

Unsupervised clustering of fixed-length windows of a polysomnography (PSG) channel, such as
nasal airflow, to surface breathing patterns (normal breathing, apnea, hypopnea, artefacts)
without labels. Models are fitted on one recording and applied to another.

## How it works

```
EDF file ──► one channel segment ──► fixed windows ──► per-window preprocessing
                                                            │
                                   smooth · detrend · 0.1-2 Hz band-pass · de-spike · z-scale
                                                            ▼
                                                   feature extraction  (cached, resumable)
                                                            ▼
                                   scale ─► select ─► reduce (fitted on training only, persisted)
                                                            ▼
                                                 clusterers (fitted on training, persisted)
                                                            ▼
                               CSV with cluster per window · PNG / HTML plots · metrics.json
```

* **Training** fits the feature transformer and every model listed in `models` and saves them.
* **Validation** loads those artefacts and only *applies* them to the validation recording, so
  validation scores are honest: nothing is re-fitted on unseen data.

## Install

```bash
conda activate PSNG            # or any Python >= 3.11 environment
pip install -r requirements.txt
```

## Run

```bash
python -m psng all                       # training then validation
python -m psng train                     # training only
python -m psng validate                  # validation only (needs a prior training run)
python -m psng all --settings configs/other.yaml
python -m psng train --no-resume         # ignore caches and refit everything
```

Run from the repository root. Relative paths in the settings file resolve against the
working directory.

## Settings (`configs/settings.yaml`)

Settings are validated with Pydantic at start-up; a typo or an unknown model / extractor name
fails immediately with a clear message.

| Section | Keys | Notes |
|---|---|---|
| `data` | `channel`, `window_seconds`, `hop_seconds`, `smoothing_seconds`, `training`, `validation` | `hop_seconds: 0` means non-overlapping windows. Each stage has `file_path`, `start_seconds`, `end_seconds` (omit `end_seconds` for the whole recording). |
| `features.extractor` | `freqtimepatt`, `basic`, `hjorth`, `wavelet`, `bandpass` | See `psng/features/`. |
| `features.scale` | bool | z-score features with statistics from training. |
| `features.selection` | `type: none | pca | correlation` | `pca` keeps features with a strong loading on the components that explain `pca_variance`; `correlation` drops one of each pair above `correlation_threshold`. |
| `features.reduction` | `type: none | pca`, `n_components` | PCA fitted on training and reused for validation. |
| `models` | `<name>: {constructor args}` | `kmeans`, `minibatch`, `bisecting`, `gmm`, `dbscan`, `kdtree`, `isolation`. See `psng/clustering/`. |
| `run` | `output_dir`, `log_dir`, `log_level`, `resume`, `n_jobs`, `chunk_windows`, `interactive_plots`, `plot_max_points` | `n_jobs: -1` uses every core for feature extraction. |

## Outputs

```
outputs/
  cache/features_<stage>_<hash>.csv      window features, keyed by a hash of the data/window/extractor settings
  training/
    models/feature_transformer.pkl       scaler + selection + PCA fitted on training
    models/<model>.pkl                   fitted clusterer
    csvs/<model>.csv                     one row per window: times, raw features, cluster_id, cluster_label
    plots/<model>.png                    2-D PCA scatter of windows and centres
    plots/<model>.html                   interactive signal coloured by cluster, per-cluster toggles
    metrics.json                         quality metrics per model
  validation/                            csvs, plots and metrics.json for the validation recording
logs/training.log, logs/validation.log
```

## Resuming

Everything expensive is checkpointed, so a run that is interrupted (or re-run after a crash)
picks up where it stopped:

* **Features** are written to the cache in chunks of `chunk_windows`. On restart the cache is
  checked row by row against the expected windows; the valid prefix is kept and extraction
  continues from the first missing window. The file name carries a hash of every setting that
  affects features, so a changed window size or extractor never reuses a stale cache.
* **Feature transformer** and **models** are reused when a saved artefact exists and its settings
  match the current ones; otherwise they are refitted.
* `--no-resume` (or `run.resume: false`) discards the caches and refits everything.

## Metrics

`metrics.json` and the log report, per model and stage:

| Metric | Meaning |
|---|---|
| `silhouette` | -1..1, higher is better. Noise windows (label -1) are excluded. Computed on at most 5000 sampled windows. |
| `calinski_harabasz` | Ratio of between- to within-cluster dispersion, higher is better. |
| `davies_bouldin` | Average similarity of each cluster to its closest cluster, lower is better. |
| `quality` | Silhouette bucket: excellent > 0.7, good > 0.5, fair > 0.25, poor > 0, else bad. |
| `n_clusters`, `n_noise`, `fit_seconds`, `resumed` | Bookkeeping. |

Compare training and validation values: a large drop on validation means the clusters do not
transfer between recordings.

## Extending

* **New extractor**: subclass `FeatureExtractor` in `psng/features/`, implement
  `extract(window) -> dict[str, float]`, register it in `EXTRACTORS`.
* **New clusterer**: subclass `Clusterer` in `psng/clustering/`, implement `fit` (set `labels_`
  and `centers_`) and `predict`, register it in `CLUSTERERS`. Saving and loading are inherited.

## Tests

```bash
python -m pytest
```

The tests run on synthetic breathing signals and cover preprocessing edge cases (flat windows,
NaNs), every extractor and clusterer, transformer persistence, metrics, the end-to-end
training / validation flow, and resuming from a partially written feature cache.

## Design notes

* Only the requested channel segment is read from the EDF; the full 26-channel recording is
  never loaded into memory.
* Per-window filters are designed once per run, not once per window.
* t-SNE and UMAP are not offered as reducers because they cannot project unseen windows, which
  makes validation meaningless.
* Density models (`dbscan`, `kdtree`, `isolation`) assign unseen windows from the fitted
  training structure (nearest core sample within `eps`, or nearest cluster mean); they never
  re-cluster the validation data.

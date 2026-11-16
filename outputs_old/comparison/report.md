# PSNG performance report

## Airflow events per recording: threshold rules and HMM

| detector | stage | duration_hours | sleep_hours | apnea | hypopnea | artefact | events_per_hour | ahi | event_window_fraction | baseline_amplitude | desaturations | odi | confirmed_fraction | ahi_confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | training | 9.42 | 5.14 | 2 | 123 | 0 | 13.3 | 14.6 | 12.4% | 48.7 | 291 | 30.1 | 74.7% | 11.1 |
| rules | validation | 9.05 | 4.51 | 21 | 157 | 1 | 19.7 | 19.3 | 18.1% | 26.9 | 252 | 27.1 | 70.1% | 14.6 |
| HMM | training | 9.42 | 5.14 | 2 | 211 | 0 | 22.6 | 22.8 | 22.4% | 48.7 | 291 | 30.1 | 67.5% | 15.6 |
| HMM | validation | 9.05 | 4.51 | 18 | 192 | 2 | 23.2 | 23.5 | 23.5% | 26.9 | 252 | 27.1 | 69.8% | 17.3 |

## Every detector against SpO2 desaturations (sleep only)

| stage | method | events_per_hour | confirmed | chance | lift | explained | f1 |
|---|---|---|---|---|---|---|---|
| training | rules, rolling median | 14.6 | 0.747 | 0.222 | 3.37 | 0.381 | 0.504 |
| training | rules, kalman | 22.9 | 0.797 | 0.224 | **3.55** | **0.548** | **0.650** |
| training | HMM, reduced-flow state | 22.8 | 0.675 | 0.231 | 2.92 | 0.542 | 0.601 |
| training | kmeans, event clusters | 10.3 | 0.849 | 0.272 | 3.12 | 0.432 | 0.573 |
| training | minibatch, event clusters | 11.1 | 0.860 | 0.272 | 3.16 | 0.484 | 0.619 |
| training | bisecting, event clusters | 6.0 | **0.903** | 0.261 | 3.46 | 0.265 | 0.409 |
| validation | rules, rolling median | 19.2 | 0.701 | 0.180 | **3.90** | 0.475 | 0.567 |
| validation | rules, kalman | 28.5 | 0.651 | 0.185 | 3.52 | **0.623** | 0.637 |
| validation | HMM, reduced-flow state | 23.4 | 0.698 | 0.186 | 3.76 | 0.598 | 0.644 |
| validation | kmeans, event clusters | 8.4 | 0.921 | 0.249 | 3.69 | 0.516 | 0.662 |
| validation | minibatch, event clusters | 8.6 | **0.923** | 0.252 | 3.67 | 0.533 | 0.676 |
| validation | bisecting, event clusters | 6.6 | 0.900 | 0.236 | 3.82 | 0.344 | 0.498 |
| validation | XGBoost on rule labels | 10.6 | 0.917 | 0.246 | 3.72 | 0.607 | **0.730** |

## HMM events against the rule-based event windows

| stage | event_f1 | event_precision | event_recall |
|---|---|---|---|
| training | 0.631 | 0.490 | 0.886 |
| validation | 0.714 | 0.631 | 0.821 |

## Clustering metrics per algorithm and stage

| algorithm | stage | n_clusters | n_noise | silhouette | davies_bouldin | calinski_harabasz | event_f1 | event_precision | event_recall | event_nmi | stage_nmi | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kmeans | training | 5 | 0 | 0.252 | **1.133** | **447.9** | **0.580** | 0.605 | 0.557 | 0.130 | **0.086** | 0.000 |
| minibatch | training | 5 | 0 | 0.251 | 1.147 | 444.0 | 0.573 | 0.576 | **0.571** | 0.129 | 0.079 | 0.000 |
| bisecting | training | 5 | 0 | 0.265 | 1.250 | 407.3 | 0.433 | **0.778** | 0.300 | **0.143** | 0.079 | 0.000 |
| gmm | training | 5 | 0 | 0.166 | 2.120 | 181.6 | 0.000 | 0.000 | 0.000 | 0.084 | 0.084 | 0.000 |
| dbscan | training | 1 | 137 | - | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| isolation | training | 2 | 81 | **0.540** | 3.163 | 50.7 | 0.000 | 0.000 | 0.000 | 0.047 | 0.023 | 0.000 |
| kmeans | validation | 5 | 0 | 0.171 | **1.138** | 233.6 | **0.707** | 0.678 | 0.740 | 0.221 | 0.109 | 0.000 |
| minibatch | validation | 5 | 0 | 0.163 | 1.183 | 227.9 | 0.707 | 0.668 | **0.750** | 0.215 | **0.120** | 0.000 |
| bisecting | validation | 5 | 0 | 0.197 | 1.401 | **295.5** | 0.663 | **0.753** | 0.592 | **0.226** | 0.098 | 0.000 |
| gmm | validation | 5 | 0 | 0.034 | 1.800 | 98.2 | 0.601 | 0.527 | 0.699 | 0.156 | 0.109 | 0.000 |
| dbscan | validation | 1 | 394 | - | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| isolation | validation | 2 | 149 | **0.446** | 1.829 | 183.9 | 0.000 | 0.000 | 0.000 | 0.153 | 0.039 | 0.000 |

## Event detection: rolling-median vs Kalman baseline

| stage | baseline | apnea | hypopnea | artefact | events_per_hour | ahi | event_window_fraction |
|---|---|---|---|---|---|---|---|
| training | rolling_median | 2 | 123 | 0 | 13.3 | 14.6 | 12.4% |
| training | kalman | 11 | 186 | 0 | 20.9 | 22.9 | 19.0% |
| validation | rolling_median | 21 | 157 | 1 | 19.7 | 19.3 | 18.1% |
| validation | kalman | 30 | 220 | 1 | 27.6 | 28.6 | 25.5% |

## Best algorithm per metric (validation)

* **Finds breathing events (event F1)**: kmeans (0.707)
* **Agrees with event labels (NMI)**: bisecting (0.226)
* **Agrees with sleep stages (NMI)**: minibatch (0.120)
* **Groups well separated (silhouette)**: isolation (0.446)
* **Groups overlap (Davies-Bouldin, lower = better)**: kmeans (1.138)
* **Groups compact (Calinski-Harabasz)**: bisecting (295.467)

## Figures

* `comparison/metrics.png` - every metric per algorithm, training next to validation
* `comparison/oximetry.png` / `oximetry_scores.png` - events against SpO2: an excerpt, and every detector scored
* `comparison/<algorithm>_card.png` - one card per algorithm (rules, clusterers, HMM, XGBoost) in the same layout
* `comparison/<algorithm>_examples.png` - example signal slices per group, or where the algorithm agrees / disagrees with the rules
* `comparison/event_rates.png` - apnea / hypopnea share per cluster
* `<stage>/plots/events.png` - airflow envelope with detected events and the hypnogram
* `<stage>/plots/<model>_timeline.png` - cluster of every window over the night, events shaded

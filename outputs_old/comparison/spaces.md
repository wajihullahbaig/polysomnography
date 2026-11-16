## Feature spaces side by side

| space | algorithm | stage | n_clusters | n_noise | silhouette | davies_bouldin | event_f1 | event_precision | event_recall | event_nmi | stage_nmi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autoencoder3 | bisecting | training | 5 | 0 | 0.227 | 1.253 | 0.000 | 0.000 | 0.000 | 0.099 | 0.051 |
| pca3 | bisecting | training | 5 | 0 | 0.265 | 1.250 | 0.433 | 0.778 | 0.300 | 0.143 | 0.079 |
| raw | bisecting | training | 5 | 0 | 0.175 | 1.785 | 0.406 | **0.809** | 0.271 | 0.147 | 0.071 |
| autoencoder3 | dbscan | training | 1 | 0 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pca3 | dbscan | training | 1 | 137 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| raw | dbscan | training | 5 | 1013 | 0.237 | **0.903** | 0.000 | 0.000 | 0.000 | 0.000 | 0.268 |
| autoencoder3 | gmm | training | 5 | 0 | 0.113 | 2.326 | 0.000 | 0.000 | 0.000 | 0.076 | 0.063 |
| pca3 | gmm | training | 5 | 0 | 0.166 | 2.120 | 0.000 | 0.000 | 0.000 | 0.084 | 0.084 |
| raw | gmm | training | 5 | 0 | 0.056 | 2.206 | 0.553 | 0.684 | 0.464 | 0.149 | 0.162 |
| autoencoder3 | isolation | training | 2 | 0 | 0.447 | 3.200 | 0.000 | 0.000 | 0.000 | 0.025 | 0.012 |
| pca3 | isolation | training | 2 | 81 | **0.540** | 3.163 | 0.000 | 0.000 | 0.000 | 0.047 | 0.023 |
| raw | isolation | training | 6 | 956 | 0.122 | 1.330 | 0.325 | 0.561 | 0.229 | **0.284** | **0.375** |
| autoencoder3 | kmeans | training | 5 | 0 | 0.275 | 1.103 | **0.599** | 0.551 | **0.657** | 0.122 | 0.067 |
| pca3 | kmeans | training | 5 | 0 | 0.252 | 1.133 | 0.580 | 0.605 | 0.557 | 0.130 | 0.086 |
| raw | kmeans | training | 5 | 0 | 0.134 | 1.734 | 0.562 | 0.695 | 0.471 | 0.128 | 0.091 |
| autoencoder3 | minibatch | training | 5 | 0 | 0.260 | 1.147 | 0.000 | 0.000 | 0.000 | 0.119 | 0.063 |
| pca3 | minibatch | training | 5 | 0 | 0.251 | 1.147 | 0.573 | 0.576 | 0.571 | 0.129 | 0.079 |
| raw | minibatch | training | 5 | 0 | 0.118 | 1.977 | 0.000 | 0.000 | 0.000 | 0.130 | 0.067 |
| autoencoder3 | bisecting | validation | 5 | 0 | 0.155 | 1.350 | 0.477 | 0.606 | 0.393 | 0.089 | 0.080 |
| pca3 | bisecting | validation | 5 | 0 | 0.197 | 1.401 | 0.663 | 0.753 | 0.592 | 0.226 | 0.098 |
| raw | bisecting | validation | 5 | 0 | 0.133 | 1.603 | 0.674 | **0.766** | 0.602 | **0.234** | 0.092 |
| autoencoder3 | dbscan | validation | 1 | 2 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pca3 | dbscan | validation | 1 | 394 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| raw | dbscan | validation | 3 | 1056 | 0.315 | 1.181 | 0.000 | 0.000 | 0.000 | 0.000 | **0.355** |
| autoencoder3 | gmm | validation | 5 | 0 | 0.001 | 2.205 | 0.000 | 0.000 | 0.000 | 0.095 | 0.107 |
| pca3 | gmm | validation | 5 | 0 | 0.034 | 1.800 | 0.601 | 0.527 | 0.699 | 0.156 | 0.109 |
| raw | gmm | validation | 5 | 0 | -0.015 | 1.766 | 0.652 | 0.627 | 0.679 | 0.207 | 0.104 |
| autoencoder3 | isolation | validation | 2 | 0 | 0.337 | 2.314 | 0.000 | 0.000 | 0.000 | 0.131 | 0.015 |
| pca3 | isolation | validation | 2 | 149 | **0.446** | 1.829 | 0.000 | 0.000 | 0.000 | 0.153 | 0.039 |
| raw | isolation | validation | 4 | 831 | -0.240 | 1.369 | 0.565 | 0.529 | 0.607 | 0.119 | 0.168 |
| autoencoder3 | kmeans | validation | 5 | 0 | 0.175 | 1.248 | 0.620 | 0.652 | 0.592 | 0.155 | 0.086 |
| pca3 | kmeans | validation | 5 | 0 | 0.171 | **1.138** | 0.707 | 0.678 | 0.740 | 0.221 | 0.109 |
| raw | kmeans | validation | 5 | 0 | 0.066 | 1.594 | **0.708** | 0.711 | 0.704 | 0.230 | 0.104 |
| autoencoder3 | minibatch | validation | 5 | 0 | 0.195 | 1.227 | 0.603 | 0.594 | 0.612 | 0.147 | 0.089 |
| pca3 | minibatch | validation | 5 | 0 | 0.163 | 1.183 | 0.707 | 0.668 | **0.750** | 0.215 | 0.120 |
| raw | minibatch | validation | 5 | 0 | 0.068 | 1.718 | 0.631 | 0.669 | 0.597 | 0.214 | 0.101 |

### Validation event F1 per algorithm and feature space

| algorithm | autoencoder3 | pca3 | raw |
|---|---|---|---|
| bisecting | 0.477 | 0.663 | 0.674 |
| dbscan | 0.000 | 0.000 | 0.000 |
| gmm | 0.000 | 0.601 | 0.652 |
| isolation | 0.000 | 0.000 | 0.565 |
| kmeans | **0.620** | **0.707** | **0.708** |
| minibatch | 0.603 | 0.707 | 0.631 |

## Supervised reference (gradient boosting on the rule-based event labels)

| backend | target | n_train | n_validation | train_f1 | validation_f1 | validation_precision | validation_recall | validation_auc |
|---|---|---|---|---|---|---|---|---|
| xgboost | event | 1130 | 1085 | 0.903 | 0.731 | 0.675 | 0.796 | 0.937 |

Top features: longest_low_flow_seconds (0.35), apnea_fraction (0.15), envelope_cv (0.09), envelope_min (0.07), low_flow_fraction (0.06), breathing_regularity (0.05)

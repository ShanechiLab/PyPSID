# Smoothing PSID Simulation Results

- Systems: `50` | Training sizes: `10,000, 100,000, 1,000,000` | Test samples: `1,000,000` | Horizon: `5` | Seed: `42`

## Parameter Identification Errors (Fig. 5, median normalized error)

| N_train | A | Cy | Cz | Sigma_xy | K | Kf | Sigma_y | Cz*Kf | eig(z) | Cz*Kf(BW) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0485 | 0.0332 | 0.0256 | 0.0425 | 0.0431 | 0.9913 | 0.0270 | 0.0392 | 0.0139 | 0.0589 |
| 100,000 | 0.0176 | 0.0101 | 0.0095 | 0.0137 | 0.0144 | 0.9990 | 0.0089 | 0.0162 | 0.0037 | 0.0233 |
| 1,000,000 | 0.0060 | 0.0038 | 0.0026 | 0.0048 | 0.0051 | 0.9999 | 0.0034 | 0.0046 | 0.0013 | 0.0178 |

## Ablation Results for Decoding $R^2$ (Table 4, $N_{\text{train}} = 1,000,000$, mean +/- SEM)

This table is equivalent to Table 4 in [the paper](https://openreview.net/pdf?id=vFZhylvjKq), using the same row ordering and model descriptions.

| Setting | Model | Mean decoding R^2 | SEM |
| --- | --- | ---: | ---: |
| *One-step-ahead prediction* | Standard PSID, used as a Kalman predictor | 0.6315 | 0.0313 |
|  | Ideal prediction (true model parameters) | 0.6325 | 0.0313 |
| *Filtering* | Standard PSID's parameters, used in a Kalman filter | 0.6315 | 0.0313 |
|  | Prediction with time-shifted data | 0.7393 | 0.0258 |
|  | **PSID with filtering (ours)** | 0.8071 | 0.0215 |
|  | Ideal filtering (true model parameters) | 0.8080 | 0.0214 |
| *Smoothing* | Standard PSID's parameters, used in a Kalman smoother | 0.6315 | 0.0313 |
|  | PSID with smoothing, backward pass on $z$ itself (non-residual) | 0.8062 | 0.0233 |
|  | **PSID with smoothing (ours, residual backward pass)** | 0.8287 | 0.0206 |
|  | Ideal smoothing (true model parameters) | 0.8300 | 0.0204 |

### Additional variants (not in Table 4)

| Model | Mean decoding R^2 | SEM |
| --- | ---: | ---: |
| PSID (updateKfKv=True), used as a Kalman predictor | 0.6315 | 0.0313 |
| PSID with smoothing, residual backward pass on $z$ only | 0.8221 | 0.0205 |

# Model: Multi-Layer Perceptron (family: Deep Learning / PyTorch) - Owner: M4 (Platform Lead)

Source files: `results/models/mlp/cv.json`, `results/models/mlp/test.json`, `results/models/mlp/history.csv`, `configs/models/mlp.yaml`, `src/pathsnap/models/mlp.py`  
Partitions used: 5-Fold Group Cross-Validation on 80 Train Trips (grouped by `trip_id`) & 10 Unseen Student-Test Trips.

---

## A. Model Understanding

### A1. Why is this algorithm suitable for PathSnap?
The Multi-Layer Perceptron (MLP) frames candidate arc scoring as a high-capacity continuous representation mapping. Given standardized 37-dimensional spatial features $\mathbf{x} \in \mathbb{R}^{37}$, the network learns non-linear feature embeddings through parameterized affine transformations and non-linear activations:
1. **Continuous Representation Learning**: Unlike axis-aligned decision trees, the MLP learns smooth, arbitrary decision boundaries across continuous features (e.g., coordinates, continuous angular deviations, distance projections).
2. **Extreme Inference Throughput**: PyTorch matrix multiplications ($\mathbf{W}\mathbf{x} + \mathbf{b}$) executed via vectorized BLAS run in **1.16 ms per 1,000 observations**—more than **6× faster than Random Forest**. This makes MLP the superior candidate for real-time edge deployment and low-latency API serving on Laptop 1.
3. **Imbalance Compensation via Loss Weighting**: Class imbalance (~1:11) is addressed directly within the optimization objective using positive class weighting (`pos_weight=4.0`) in the binary cross-entropy loss, ensuring strong gradient flow for true candidate arcs.
4. **Smooth Posterior Calibration**: The output layer applies a standard Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, providing calibrated emission likelihoods required by M3's downstream HMM Viterbi decoder.

### A2. Major Strengths and Limitations for This Task
- **Strengths**:
  - Ultra-fast inference: **1.16 ms / 1,000 obs**, fulfilling strict production latency SLAs.
  - High accuracy: Matches ensemble performance with **89.42% Top-1 observation accuracy** and **0.9357 Macro-F1** on unseen student-test trips.
  - Lightweight storage: Model weights (`model.joblib`) take only 25 KB, orders of magnitude smaller than tree ensembles.
- **Limitations**:
  - Feature Scale Sensitivity: Requires strict feature standardization (`StandardScaler` stored in `artifacts/scaler.joblib`); unnormalized inputs degrade convergence.
  - Training Time: Requires iterative mini-batch backpropagation (84.99s full fit, ~67.16s/fold) compared to fast single-pass tree building.

---

## B. Training and Configuration

### B5. Important Hyperparameters and Initial Values

| Hyperparameter | Value | Rationale |
|:---|:---|:---|
| Architecture | `[37, 64, 32, 1]` | 2 hidden layers with bottleneck compression for compact feature representation |
| Activations | ReLU | Mitigates vanishing gradients across hidden layers |
| Normalization | `BatchNorm1d` (64, 32) | Stabilizes internal covariate shift during mini-batch gradient descent |
| Regularization | `Dropout(0.1)` + $L_2$ (`weight_decay=1e-4`) | Prevents co-adaptation of hidden units on spatial features |
| Learning Rate | `0.001` (Adam Optimizer) | Standard adaptive learning rate with momentum ($\beta_1=0.9, \beta_2=0.999$) |
| Batch Size | 512 | Balances GPU/CPU vectorization efficiency with stochastic gradient noise |
| Epochs | 15 | Sufficient iterations for loss convergence without over-fitting validation loss |
| `pos_weight` | 4.0 | Re-weights positive arc candidate loss to counter 1:11 negative class dominance |
| Feature Scaler | `StandardScaler` | Zero-mean, unit-variance normalization fitted exclusively on training set |
| Random Seed | 42 | Full determinism in weight initialization and batch shuffling |

### B6. Loss Function / Optimisation Objective
The network optimizes a **Weighted Binary Cross-Entropy (BCE) Loss**:
$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ w \cdot y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \lambda \|\mathbf{W}\|_2^2$$
where $w = 4.0$, $\lambda = 10^{-4}$, and $\sigma(z) = \frac{1}{1 + \exp(-z)}$.

---

## C. Experimental Results

### C7. Performance on the Student-Testing Dataset
Evaluated on 10 unseen student-test trips (220,054 candidates, 19,141 GPS observations):
- **Observation Top-1 Accuracy**: **89.42%** (surpassing Random Forest's 89.38%)
- **Macro-F1**: **0.9357**
- **Binary Accuracy**: **98.00%**
- **Binary Precision / Recall**: **90.19% / 86.35%**
- **Log Loss**: **0.0671**
- **Inference Latency**: **1.16 ms per 1,000 observations** (6.1× faster than RF)

Across 5-fold cross-validation on 80 training trips, Folds 1–4 scored between 81.54% and 92.42% Top-1, with Fold 0 at 49.93% due to dense urban candidate ambiguity.

### C8. Summary Metrics Table (Student-Test)

| Metric | Value | Applicable / Reason |
|:---|:---|:---|
| Top-1 Observation Accuracy | **89.42%** | Primary evaluation metric (Argmax probability match) |
| Top-1 (Given Candidate Hit) | **89.42%** | Conditional candidate recall |
| Macro-F1 | **0.9357** | Harmonic mean of precision/recall across both classes |
| Binary Precision | **90.19%** | $16,529 / (16,529 + 1,798)$ |
| Binary Recall | **86.35%** | $16,529 / (16,529 + 2,612)$ |
| Log Loss | **0.0671** | Cross-entropy calibrated probability loss |
| Inference Latency | **1.16 ms / 1k obs** | Measured over 19,141 test observations |
| Optimal Threshold ($\tau^*$) | **0.800** | Selected via out-of-fold F1 optimization |

### C9. Overfitting / Underfitting Assessment
- **Evidence**:
  - Out-of-fold Aggregate Macro-F1 was **0.9207** (std: 0.0448). On unseen student-test data, Macro-F1 increased to **0.9357**, showing excellent generalizability.
  - **Real Epoch Loss Curves** (`results/models/mlp/history.csv`):
    - Epoch 1: Train Loss = 0.1287, Validation Loss = 0.0941
    - Epoch 5: Train Loss = 0.0923, Validation Loss = 0.0844
    - Epoch 10: Train Loss = 0.0889, Validation Loss = 0.0808
    - Epoch 15: Train Loss = 0.0875, Validation Loss = 0.0803
    Both training and validation losses smoothly decay and asymptote without the validation loss diverging upward, confirming that BatchNorm, Dropout (0.1), and $L_2$ weight decay successfully prevented overfitting.

### Visualizations
- **Confusion Matrix**: `results/models/mlp/confusion_matrix.png` (TN: 199,115, FP: 1,798, FN: 2,612, TP: 16,529)
- **Loss Curve**: `results/models/mlp/loss_curve.png` (Real epoch train vs. validation loss trajectory)

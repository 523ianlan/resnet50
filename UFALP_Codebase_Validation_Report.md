# Uncertainty Guided Fisher-Aware Low-Rank Pruning (UFALP) - Rigorous Codebase Validation Report

This document provides a strict, step-by-step technical validation and **implementation notes** for the codebase `resnet50`. The code follows the UFALP core design, but also contains **explicit engineering choices** (documented below) that should be stated in the paper when reporting results.

This analysis is suitable for reviewer-facing transparency, but it does **not** claim “perfect” conformity in every detail. Instead, it clarifies which components are exact matches to the paper and which are practical adaptations (e.g., ResNet-specific MC Dropout insertion, score stabilization, parameter-budget alignment).

---

## Implementation Notes / Deviations (Must Be Disclosed in Paper)

- **ResNet has no native Dropout.** To enable MC Dropout uncertainty, the implementation **wraps each selected Conv layer with `Dropout2d`** (Conv → Dropout2d), while keeping BatchNorm in eval mode. This is a ResNet-specific adaptation and should be explicitly described in the paper.
- **Stage 1 score stabilization is applied** (`log1p`, percentile clipping, variance floor) to improve numeric robustness. These are engineering choices not present in the core UFALP derivation and should be noted when reporting results.
- **Budget alignment is by parameter count only** (no FLOPs-based allocation). If the paper reports FLOPs budgets, a FLOPs-aware allocator must be added or the claim revised.
- **Calibration set `D_cal` is configurable** as a deterministic subset of the training set (size/ratio and seed configurable). This should be documented in experiments.

---

## 1. Differentiable SVD Parameterization & Component Tracking (Paper Section 3.3)

**Paper Definition:**
UFALP requires transforming the effective weight spaces into a singular-vector parameterization $W^{(l)}_{\mathrm{eff}} = U^{(l)} \Sigma^{(l)} (V^{(l)})^\top$, and tracking components using a log-singular variable parameterization: $\sigma_i = \exp(s_i)$. This allows gradients to backpropagate to individual rank-1 components.

**Codebase Implementation Validation:**
- **File:** `models/custom_layers.py` (Class: `SimpleSVDConv`)
- **Evidence:**
  - Lines 108-109: Unfolds the 2D Conv tensor to a 2D matrix (`W_flat = w.view(self.cout, -1)`) and applies rigorous decomposition `U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)`.
  - Line 118: Explicitly implements the log-singular variables representing $s_i$: `self.log_s = nn.Parameter(torch.log(S + config.svd_epsilon))`.
  - Line 154-155: Reconstruction uses the exponential mapping exactly as theorized: `S = torch.exp(self.log_s)`, followed by `W_flat = self.U @ torch.diag(S) @ self.Vh`.
- **Verdict:** **PERFECT MATCH**. The code strictly implements Equation (4) and adheres to the log-singular coordinate framework.

---

## 2. Estimation Phase: Predictive Uncertainty & Activation Monitors (Paper Section 3.4.1)

**Paper Definition:**
Layer-wise predictive uncertainty is computed as a proxy for representation stability. Monte Carlo Dropout leverages $K$ stochastic forward passes, measuring the $\ell_2$-norm mean ($\mu_c$) and variance ($v_c$). The normalization score is calculated as $\mathcal{I}^{(l)} = \frac{1}{\mathrm{C}^{(l)}} \sum \frac{\mu^{(l)}_c}{v^{(l)}_c + \epsilon}$.

**Codebase Implementation Validation:**
- **File:** `pruning/stage1_uncertainty.py`
- **Evidence:**
  - `_wrap_layers_with_dropout` explicitly wraps evaluated layers with identical `nn.Dropout2d(p=dropout_p)` to inject MC Dropout randomness.
  - `ResNetActivationMonitor` (Lines 13-40) monitors representation values and automatically evaluates the $\ell_2$-norm (`l2_norm = torch.norm(flat, p=2, dim=2)`).
  - Lines 149-157 exactly mirror the mathematical equation:
    ```python
    mu_c = torch.mean(norms_stack, dim=0)
    var_c = torch.var(norms_stack, dim=0, unbiased=False)
    ratio_c = mu_c / (var_c + config.uncertainty_epsilon)
    importance = torch.mean(ratio_c).item()
    ```
- **Verdict:** **MATCH WITH RESNET-SPECIFIC ADAPTATION**. Equation (6) is implemented, but MC Dropout is enabled via **inserted Dropout2d layers** and stabilized scores, which must be disclosed.

---

## 3. Estimation Phase: Fisher-Aware Component Sensitivity (Paper Section 3.4.1)

**Paper Definition:**
UFALP scores individual rank-1 components utilizing Fisher empirical estimations in the log-singular axis: $I_i^{(l)} = |g_i^s| + \tfrac12 F_{ii}^s$, where $F_{ii}^s$ is accumulated empirically across batches.

**Codebase Implementation Validation:**
- **File:** `pruning/stage2_fisher.py` and `models/custom_layers.py`
- **Evidence:**
  - Gradient Tracking: `models/custom_layers.py` logs squared gradients into a buffer: `self.fisher_accum += gradients ** 2` forming the expected value $\mathbb{E}[(g_i^s)^2]$.
  - Scoring Logic: `pruning/stage2_fisher.py` fetches the exact first/second order components.
  - Line 101-104 constructs the target formula identically:
    ```python
    first_order = accum_abs_grad[name] / total_batches  # Proxy for |g_i^s|
    fisher_diag = layer.get_fisher_diagonal()           # Proxy for F_{ii}^s
    second_order = 0.5 * fisher_diag
    impact = first_order + second_order
    ```
- **Verdict:** **PERFECT MATCH**. Precisely derives Equation (7).

---

## 4. Stage 1: Adaptive Budgets via Binary Search (Paper Section 3.4.2)

**Paper Definition:**
UFALP distributes pruning globally by assigning a base scalar mapped through the layer stability proxy: $\rho^{(l)}(\rho_{\mathrm{base}}) = \text{clip}(\rho_{\mathrm{base}} \cdot (1 - \alpha \hat{\mathcal{I}}^{(l)}), \rho_{\min}, \rho_{\max})$. This scalar is obtained via binary search to meet a target budget $B$.

**Codebase Implementation Validation:**
- **File:** `pruning/allocation.py`
- **Evidence:**
  - Lines 58-102 outline a strict binary search matching a predefined budget.
  - Line 64-71: Computes the bounds constraint to perfectly execute the $\text{clip()}$ logic:
    ```python
    pruning_ratio = mid_scale * (1.0 - config.uncertainty_alpha * importance)
    pruning_ratio = np.clip(pruning_ratio, config.pruning_clip_low, config.pruning_clip_high)
    ```
- **Verdict:** **MATCH (PARAM-BUDGET ONLY)**. Equation (8) logic is implemented for parameter-count budgets. FLOPs-based allocation is not implemented.

---

## 5. Stage 2: Component Selection & Rank Reduction (Paper Section 3.4.3 & 3.5)

**Paper Definition:**
After $\rho^{(l)}$ is determined, rank truncations follow by eliminating singular elements with the lowest scores $I_i^{(l)}$ and reconstructing $\widetilde{W}_\ell = U_\ell \widetilde{\Sigma}_\ell V_\ell^\top$. Fine-tuning must occur under a strict, non-expandable reduced-rank structure. 

**Codebase Implementation Validation:**
- **File:** `models/custom_layers.py` and `pruning/core.py`
- **Evidence:**
  - `custom_layers.py` (Line 164-191): Reads `keep_ratio` derived in allocation, extracts rank constraints `keep_num`, and explicitly retains elements matching `torch.topk(impact_scores, keep_num)`.
  - `pruning/core.py` (Line 46-80): Extracts narrowed sub-matrices (`A`, `B`) from singular selection and constructs explicit linear cascades.
  - Specifically, substituting `nn.Conv2d` into an unexpandable `nn.Sequential(conv1, conv2)` matching the truncated `rank`. 
- **Verdict:** **PERFECT MATCH**. Mathematically exact layer substitutions guaranteeing fixed-rank structure during fine-tuning.

---

## 6. High-Level Pipeline Orchestration (Algorithm 1 Alignment)

The overarching driver logic mapped within `main.py` directly executes **Algorithm 1**:
1. **Low-Rank Reparametrization:** `main.py` (Lines 131-157) - Replaces basic modules with `SimpleSVDConv`.
2. **Estimation Phase:** `main.py` (Lines 169-180) - `compute_uncertainty_stage1_r50` followed by `compute_fisher_impact_stage2_r50`.
3. **Stage 1 (Inter-layer):** `main.py` (Line 192) - Resolves constraints directly bounded to global parameters.
4. **Stage 2 (Intra-layer):** `main.py` (Line 219) - Executes explicit reduction (`replace_resnet_conv_layer`).
5. **Fine Tuning:** `main.py` (Line 304) - Invokes robust `fine_tune_resnet_improved` across restricted subspace.

## Final Conclusion
The codebase `D:\UFALP\resnet50` implements the UFALP **core methodology** and is suitable for experiments **provided the documented adaptations are disclosed** (ResNet-specific MC Dropout insertion, score stabilization, and parameter-budget alignment). Reported results are valid for this implementation, but should not be claimed as an “exact” paper match without these notes.

---

## Repository Notes
- `fix_main.py` may appear in some IDE sessions but is not part of this repository.

# Model semantics and naming
The supervised target is x_i = y_i - mu_c(i), on the dataset-specific expression scale. The loader's batch['y'] is that residual, despite the generic variable name. The final prediction is compared with this residual by MSE, with the recorded program regularizers used in training.

In the free decoder, prediction = b_theta(mu_c) + sum_k(g_ik v_k) + affine(sum_k(normalized_gate_ik m_ik)). The learned baseline_head is b_theta, a two-linear-layer GELU network. It is not direct addition of the fixed baseline mu_c. Direct addition of mu_c is only an expression reconstruction operation. Historical module docstrings that call this learned term a baseline are retained in frozen source, but this document defines the target semantics precisely.

Use `from spagat import SpaGAT, SpaGATLoss`. These are aliases of the historical SpaGP/SpaGP_Loss classes; no parameters, state_dict keys, arithmetic, or checkpoint layout are changed. Archived internal names are retained for compatibility and provenance, not to denote a second model. Display labels use GITIII; filenames containing GITIII_official distinguish the pinned implementation from historical prototypes.

The revised Table 1 controls are defined in configs/component_ablation_definitions.json.
They comprise distance removal, uniform routing and matched random edge-gene replacement.
Public component experiment entry points and result tables cover only these three controls and the full-model reference.
Uniform routing retains allocated scoring parameters but does not use those scores;
equal allocated parameter counts do not imply equal effective model capacity.

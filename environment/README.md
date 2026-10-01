# Recorded environments
Mouse_recorded.json and SEA_AD_recorded.json preserve the benchmark runtime reports (Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3, pandas 2.2.3, LightGBM 4.6.0, torch-geometric 2.8.0.post1; CUDA 12.8 / Tesla T4 where recorded).

component_ablation_full_freeze.txt is the saved package freeze for the completed component ablations. component_ablation_selected_pins.txt extracts only directly relevant packages. These are observed environment records, not a claim that a fresh installation has been validated. Colab-specific or platform-specific entries in the full freeze should not be blindly installed on macOS.

requirements-plots.txt is a lightweight convenience dependency list for table/figure regeneration; it is not the training environment. The current assembly checks use the local environment reported in validation_report.json, which differs from the training environment. Original liver runtime details are not sufficiently complete here for an exact environment reconstruction.

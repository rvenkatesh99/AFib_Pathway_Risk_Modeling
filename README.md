# AFib Pathway Risk Modeling

Pathway-based machine learning models for atrial fibrillation risk prediction. Models learn over a matrix of biological pathway feature scores (individuals × pathways), optionally combined with a polygenic risk score (PRS) and clinical covariates.

## Modeling
Baselines: PRS logistic regression, L1 logistic regression, random forest
Global softmax attention model (population-level pathway ranking)
Pathway transformer with CLS token and per-individual attention
GraphSAGE GNN over a pathway interaction graph

## Input data format

**Pathway matrix**: rows = individuals, columns = pathway scores. Optionally includes a `sample_id` column. A 2D matrix (N × K, one score per pathway) is the expected format; a 3D format (N × K × T, multiple features per pathway) is also supported.

**Covariates**: must include the outcome column (e.g. `afib`), age, sex, and ancestry PCs. PRS, if available, is included here as a covariate — it is passed through a separate encoder branch rather than treated as a pathway feature, keeping the pathway importance ranking interpretable on its own.

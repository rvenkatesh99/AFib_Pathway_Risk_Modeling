# AFib Pathway Risk Modeling

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia and a major cause of cardioembolic stroke. Although polygenic risk scores (PRS) are well characterized to quantify inherited susceptibility for AF, they provide limited insight into the pathways and tissues underlying genetic risk, which are critical to uncover for individual risk prediction. In this study, we develop a pathway-level multi-omics representation learning framework that converts individual genetic profiles into interpretable biological features by integrating GWAS-derived pathway burden scores with tissue-specific transcriptomic pathway signals. We constructed machine learning models to assess population-level AF risk prediction performance across genomic and transcriptomic tissue contexts; the pathway-based global attention models substantially improved risk prediction performance over PRS and other baselines (AUROC improved from 0.601 to 0.738). Transformer and graph neural network frameworks then assessed individual-level pathway interpretability, revealing heterogeneous contributions from electrical signaling, cardiac development, and DNA repair pathways to AF risk. This added interpretability highlights the potential of this pathway approach to enable more mechanistically informed risk stratification than static PRS by capturing underlying heterogeneity. To independently assess whether prioritized pathways reflected cardiac regulatory biology, we compared pathway rankings with transcriptional effects predicted by the AlphaGenome foundation model. Variants in highly ranked pathways showed significantly greater predicted effects on expression in atrial and ventricular tissues (FDR = 0.032) relative to controls, providing orthogonal evidence that the model identifies biologically relevant mechanisms. Overall, this work reframes polygenic risk from a single measure of susceptibility to tissue-informed pathway mechanisms, providing a framework for interpretable genomic stratification in complex diseases.

**This github repository includes code and data utilized in manuscript submission for PSB 2027**

**Supplemental Data files are available for download in the Supplemental Data folder. (Please download raw files; github render may not work on all devices)**

## Modeling
Baselines: Unregularized logistic regression, L1 (LASSO) logistic regression, random forest

Global softmax attention model (population-level pathway ranking)

Pathway transformer with CLS token and per-individual attention

GraphSAGE GNN over a pathway interaction graph

## Input data format

**Pathway matrix**: rows = individuals, columns = pathway scores. Optionally includes a `sample_id` column. A 2D matrix (N × K, one score per pathway) is the expected format; a 3D format (N × K × T, multiple features per pathway) is also supported.

**Covariates**: must include the outcome column (e.g. `afib`), age, sex, and ancestry PCs. PRS, if available, is included here as a covariate — it is passed through a separate encoder branch rather than treated as a pathway feature, keeping the pathway importance ranking interpretable on its own.

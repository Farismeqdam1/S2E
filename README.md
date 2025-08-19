# Multimodal AI Pipeline for Reclassifying VUS in MS-Associated Genes

This repository provides a **prototype Python pipeline** to integrate structure-aware predictors (e.g., AlphaMissense) and sequence-to-expression models (e.g., Enformer/Basenji) for classifying **Variants of Uncertain Significance (VUS)** in **Multiple Sclerosis (MS)-associated genes**.

---

## Features
- **Input support**: CSV or VCF variant files.
- **Feature integration**:
  - **AlphaMissense scores** (gene + amino acid change).
  - **Sequence-to-expression (S2E)** predictions (chrom, pos, ref, alt).
  - Allele frequency, variant consequence.
- **Machine Learning**:
  - Train ensemble classifiers (Random Forest or Logistic Regression).
  - Evaluate with ROC-AUC and PR-AUC.
  - Score new VUS with trained models.
- **Output**: Ranked variants with `pathogenicity_prob` and `predicted_label`.

---

## Installation
```bash
# Clone repository
git clone <repo-url>
cd <repo>

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- Python 3.9+
- pandas, numpy, scikit-learn, joblib
- (optional) cyvcf2 for VCF input

---

## Usage

### 1. Training a Model
Provide a CSV of labeled variants (`label` column: 1=pathogenic, 0=benign):
```bash
python ms_vus_multimodal_pipeline.py train \
  --labeled labeled_variants.csv \
  --alpha-missense alpha_missense_scores.csv \
  --s2e s2e_predictions.csv \
  --model-out model.joblib \
  --model rf
```

### 2. Scoring VUS
Apply a trained model to new variants (CSV or VCF):
```bash
python ms_vus_multimodal_pipeline.py score \
  --input vus_to_score.csv \
  --alpha-missense alpha_missense_scores.csv \
  --s2e s2e_predictions.csv \
  --model-in model.joblib \
  --out scored_vus.csv
```

---

## Input Formats
- **Variants CSV** must include: `chrom, pos, ref, alt`. Optional: `gene, consequence, aa_change, af`.
- **AlphaMissense CSV**: `gene, aa_change, alpha_missense_score`.
- **S2E CSV**: `chrom, pos, ref, alt, s2e_delta_expression`.

---

## Output
- `scored_vus.csv` includes all input columns plus:
  - `alpha_missense_score`
  - `s2e_delta_expression`
  - `pathogenicity_prob` (0–1)
  - `predicted_label` (0=benign, 1=pathogenic)

Variants are ranked by pathogenicity probability.

---

## Example Workflow
1. Collect **VUS** from MS gene panels / GWAS.
2. Annotate with **AlphaMissense** and **S2E** predictions.
3. Train model using labeled set (ClinVar, curated MS datasets).
4. Score new VUS and generate ranked outputs.
5. Prioritize top candidates for **wet-lab validation** (CRISPRi, MPRA, protein assays).

---

## License
This code is provided as a **prototype research tool**. Not for clinical use.

---

## Citation
If you use this pipeline, please cite:
> *Prototype AI framework for integrating structure- and expression-aware predictors to reclassify Variants of Uncertain Significance in Multiple Sclerosis genes.*

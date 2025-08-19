#!/usr/bin/env python3
"""
Multimodal AI Pipeline for Reclassifying VUS in MS-Associated Genes
------------------------------------------------------------------
Prototype that:
  • Ingests variants (CSV or VCF)
  • Integrates AlphaMissense + S2E features
  • Trains an ensemble classifier (Random Forest/Logistic Regression)
  • Scores VUS with saved model and outputs ranked list
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# -----------------------------
# Helpers
# -----------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df

def read_variants_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize(df)

def load_alpha_missense(path: Optional[Path]) -> Dict[Tuple[str,str], float]:
    if not path: return {}
    df = _normalize(pd.read_csv(path))
    return {(r['gene'], r['aa_change']): r['alpha_missense_score'] for _,r in df.iterrows()}

def load_s2e(path: Optional[Path]) -> Dict[Tuple[str,int,str,str], float]:
    if not path: return {}
    df = _normalize(pd.read_csv(path))
    return {(r['chrom'], int(r['pos']), r['ref'], r['alt']): r['s2e_delta_expression'] for _,r in df.iterrows()}

def add_features(df: pd.DataFrame, alpha_map, s2e_map) -> pd.DataFrame:
    df = df.copy()
    df['alpha_missense_score'] = df.apply(lambda r: alpha_map.get((r.get('gene'), r.get('aa_change')), np.nan), axis=1)
    df['s2e_delta_expression'] = df.apply(lambda r: s2e_map.get((str(r['chrom']), int(r['pos']), str(r['ref']), str(r['alt'])), np.nan), axis=1)
    return df

def build_features(df: pd.DataFrame):
    num_feats = ['alpha_missense_score','s2e_delta_expression','af']
    cat_feats = ['consequence']
    for c in num_feats:
        if c not in df: df[c] = np.nan
    for c in cat_feats:
        if c not in df: df[c] = np.nan
    return df[num_feats+cat_feats], num_feats, cat_feats

# -----------------------------
# Pipeline
# -----------------------------

def make_pipeline(num_feats, cat_feats, model='rf') -> Pipeline:
    num_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))])
    cat_trans = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    pre = ColumnTransformer([
        ('num', num_trans, num_feats),
        ('cat', cat_trans, cat_feats)])

    if model=='rf':
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
    else:
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced')

    return Pipeline([('pre', pre), ('clf', clf)])

# -----------------------------
# Train / Score
# -----------------------------

def train(labeled: Path, alpha: Path, s2e: Path, model_out: Path, model_kind='rf'):
    df = read_variants_csv(labeled)
    alpha_map, s2e_map = load_alpha_missense(alpha), load_s2e(s2e)
    df = add_features(df, alpha_map, s2e_map)
    X, num, cat = build_features(df)
    y = df['label'].astype(int)
    pipe = make_pipeline(num, cat, model_kind)
    pipe.fit(X,y)
    probs = pipe.predict_proba(X)[:,1]
    print("ROC-AUC", roc_auc_score(y, probs))
    print("PR-AUC", average_precision_score(y, probs))
    print(classification_report(y,(probs>=0.5).astype(int)))
    joblib.dump({'pipeline':pipe,'num':num,'cat':cat}, model_out)

def score(inp: Path, alpha: Path, s2e: Path, model_in: Path, out: Path):
    df = read_variants_csv(inp)
    bundle = joblib.load(model_in)
    alpha_map, s2e_map = load_alpha_missense(alpha), load_s2e(s2e)
    df = add_features(df, alpha_map, s2e_map)
    X,_,_ = build_features(df)
    pipe: Pipeline = bundle['pipeline']
    probs = pipe.predict_proba(X)[:,1]
    df['pathogenicity_prob']=probs
    df['predicted_label']=(probs>=0.5).astype(int)
    df.sort_values('pathogenicity_prob',ascending=False).to_csv(out,index=False)
    print(f"Scored {len(df)} variants -> {out}")

# -----------------------------
# CLI
# -----------------------------

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd',required=True)
    pt=sub.add_parser('train'); pt.add_argument('--labeled',type=Path,required=True); pt.add_argument('--alpha',type=Path); pt.add_argument('--s2e',type=Path); pt.add_argument('--model-out',type=Path,default=Path('model.joblib')); pt.add_argument('--model',choices=['rf','logreg'],default='rf')
    ps=sub.add_parser('score'); ps.add_argument('--input',type=Path,required=True); ps.add_argument('--alpha',type=Path); ps.add_argument('--s2e',type=Path); ps.add_argument('--model-in',type=Path,required=True); ps.add_argument('--out',type=Path,default=Path('scored_vus.csv'))
    a=p.parse_args()
    if a.cmd=='train': train(a.labeled,a.alpha,a.s2e,a.model_out,a.model)
    else: score(a.input,a.alpha,a.s2e,a.model_in,a.out)

if __name__=='__main__':
    main()

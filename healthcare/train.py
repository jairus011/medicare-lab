"""Fixed protocol: profile-group holdout, training-only CV, then one test evaluation."""

import argparse
import importlib.metadata
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data import SEED, SYMPTOMS, PROFILE_FIELDS, EXCLUDED_FIELDS, MIN_CLASS_PROFILES
from .data import prepare_data, binary_features, sha256_file, write_json


def candidates():
    # Fixed before reading the test results. Order breaks an exact CV tie by simplicity.
    models = {
        "Dummy prior": DummyClassifier(strategy="prior"),
        "Bernoulli Naive Bayes": BernoulliNB(alpha=1.0, fit_prior=False),
        "Logistic regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=SEED),
        "Decision tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
        "K nearest neighbors": KNeighborsClassifier(n_neighbors=7, weights="uniform", metric="manhattan"),
        "Random forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=1),
    }
    # Binary features need neither scaling nor imputation. Conversion is stateless.
    return {name: Pipeline([("binary", FunctionTransformer(validate=True, feature_names_out="one-to-one")), ("classifier", model)]) for name, model in models.items()}


def ranked_labels(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        scores = model.decision_function(X)
    # Stable class order resolves equal scores deterministically.
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.asarray(model.classes_)[order]


def measure(y, pred, ranked, labels):
    present = sorted(set(y))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y, pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "top3_accuracy": float(np.mean([actual in row[:3] for actual, row in zip(y, ranked)])),
        "rows": int(len(y)), "labels_present": len(present),
    }


def bootstrap_accuracy(y, pred, groups, repeats=2000):
    rng = np.random.default_rng(SEED)
    group_ids = np.unique(groups)
    buckets = [np.flatnonzero(groups == group) for group in group_ids]
    correct = np.asarray(y) == np.asarray(pred)
    samples = []
    for _ in range(repeats):
        idx = np.concatenate([buckets[i] for i in rng.integers(0, len(buckets), len(buckets))])
        samples.append(float(correct[idx].mean()))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def audit_data(raw, clean, study, scope, source_path):
    groups = raw.groupby(list(SYMPTOMS))["disease"].nunique()
    clean_counts = clean.groupby([*SYMPTOMS, "disease"]).size()
    study_counts = study.groupby([*SYMPTOMS, "disease"]).size()
    conflict = clean.groupby("profile_group")["disease"].nunique()
    return {
        "source_sha256": sha256_file(source_path),
        "raw_rows": len(raw), "raw_columns": len(raw.columns)-1,
        "raw_labels": int(raw.disease.nunique()), "missing_cells": int(raw.isna().sum().sum()),
        "duplicate_rows_removed": len(raw)-len(clean), "clean_rows": len(clean),
        "conflicting_full_profile_groups": int((conflict > 1).sum()),
        "symptom_patterns": len(groups), "ambiguous_symptom_patterns": int((groups > 1).sum()),
        "max_labels_per_symptom_pattern": int(groups.max()),
        "deterministic_symptom_ceiling_clean": float(clean_counts.groupby(level=list(range(4))).max().sum()/len(clean)),
        "deterministic_symptom_ceiling_study": float(study_counts.groupby(level=list(range(4))).max().sum()/len(study)),
        "ceiling_note": "Descriptive in-sample bound on deterministic symptom-only prediction, not validation or a clinical statistic.",
        "risk_values_per_disease": {str(k): int(v) for k,v in raw.groupby("disease").risk_level.nunique().value_counts().items()},
        "included_labels": int(scope.included.sum()), "excluded_labels": int((~scope.included).sum()),
        "study_rows": len(study), "excluded_clean_rows": len(clean)-len(study),
        "minimum_class_profiles": MIN_CLASS_PROFILES, "features": list(SYMPTOMS),
        "excluded_features": EXCLUDED_FIELDS,
        "truncated_labels": sorted(label for label in raw.disease.unique() if "..." in label or label.count("(") != label.count(")")),
        "grouping_note": "Group key hashes the 8 raw pre-outcome fields, without the target. No patient ID, time, site or clinical provenance was supplied. Independence cannot be verified.",
        "scope_note": "A predefined minimum of 5 distinct raw profiles per original label defines this small study. No Other class and no guessed alias merges. Out-of-scope conditions cannot be detected by the model.",
    }


def make_plots(reports, comparison, cm, labels, scope):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(9,4.8), layout="constrained")
    rows = comparison.sort_values("cv_macro_f1_mean")
    ax.barh(rows.model, rows.cv_macro_f1_mean, xerr=rows.cv_macro_f1_std, color="#087f8c", capsize=3)
    ax.set(xlabel="Training CV macro F1 (mean ± fold SD)", title="Classifier comparison | four grouped folds")
    ax.set_xlim(0, max(0.15, float((rows.cv_macro_f1_mean+rows.cv_macro_f1_std).max())*1.2))
    fig.savefig(reports/"figures/model_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,11), layout="constrained")
    im=ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set(xlabel="Predicted label", ylabel="Supplied label", title=f"Untouched holdout | counts, all {len(labels)} configured labels")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i,j]: ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=7,color="white" if cm[i,j]>cm.max()/2 else "#17334d")
    fig.colorbar(im, ax=ax, shrink=.6)
    fig.savefig(reports/"figures/confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax=plt.subplots(figsize=(9,5.5), layout="constrained")
    support=scope.loc[scope.included].sort_values("unique_profiles")
    ax.barh(support.condition, support.unique_profiles, color="#17334d")
    ax.tick_params(axis="y",labelsize=7)
    ax.set(xlabel="Distinct raw profiles per label",title="Included condition coverage")
    fig.savefig(reports/"figures/class_support.png",dpi=160); plt.close(fig)


def run_experiment(root: Path, output_root: Path | None = None):
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root
    reports, models = output/"reports", output/"models"
    (reports/"figures").mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    source_path=root/"data/raw/Cleaned_Dataset.csv"
    raw, clean, study, scope = prepare_data(source_path)
    X, y, groups = binary_features(study), study.disease, study.profile_group
    labels=sorted(y.unique())
    # First of five fixed folds is the only holdout. Never search for a flattering seed.
    outer=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx=next(outer.split(X,y,groups))
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]): raise AssertionError("Group leakage")
    if set(y.iloc[train_idx]) != set(labels): raise ValueError("A scoped label has no training examples; revise the protocol before evaluation")
    Xtr, ytr, gtr=X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    Xtest, ytest, gtest=X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]
    inner=StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED+1)
    folds=list(inner.split(Xtr,ytr,gtr))
    catalog=candidates(); rows=[]; details=[]
    fold_assignment={}
    for fold, (fit,valid) in enumerate(folds):
        if set(gtr.iloc[fit]) & set(gtr.iloc[valid]): raise AssertionError("CV group leakage")
        for pos in valid: fold_assignment[int(study.iloc[train_idx[pos]].source_row)]=fold
    for preference,(name,template) in enumerate(catalog.items()):
        records=[]
        for fold,(fit,valid) in enumerate(folds):
            model=clone(template).fit(Xtr.iloc[fit], ytr.iloc[fit])
            pred=model.predict(Xtr.iloc[valid]); ranked=ranked_labels(model,Xtr.iloc[valid])
            m=measure(ytr.iloc[valid],pred,ranked,labels)
            records.append(m); details.append({"model":name,"fold":fold,**m,"training_labels":len(model.classes_)})
        rows.append({"model":name,"preference":preference, **{f"cv_{key}_{agg}":float(getattr(np,agg)([m[key] for m in records])) for key in ("macro_f1","accuracy","balanced_accuracy","top3_accuracy") for agg in ("mean","std")}})
    comparison=pd.DataFrame(rows).sort_values(["cv_macro_f1_mean","cv_balanced_accuracy_mean","preference"],ascending=[False,False,True]).reset_index(drop=True)
    chosen=str(comparison.iloc[0].model)
    selected=clone(catalog[chosen]).fit(Xtr,ytr)
    pred=selected.predict(Xtest); ranked=ranked_labels(selected,Xtest)
    final=measure(ytest,pred,ranked,labels)
    baseline=clone(catalog["Dummy prior"]).fit(Xtr,ytr)
    dummy=measure(ytest,baseline.predict(Xtest),ranked_labels(baseline,Xtest),labels)
    # A separate challenge to unseen 4-bit patterns. Uses TRAINING ONLY; not for selection.
    pattern_groups=Xtr.astype(str).agg("".join,axis=1)
    stress=[]
    for fold,(fit,valid) in enumerate(GroupKFold(n_splits=4).split(Xtr,ytr,pattern_groups)):
        model=clone(catalog[chosen]).fit(Xtr.iloc[fit],ytr.iloc[fit])
        m=measure(ytr.iloc[valid],model.predict(Xtr.iloc[valid]),ranked_labels(model,Xtr.iloc[valid]),labels)
        stress.append({"fold":fold,**m,"unseen_labels":sorted(set(ytr.iloc[valid])-set(model.classes_))})
    # Save the exact model evaluated on the holdout; no post-test full-data refit.
    joblib.dump(selected,models/"model.joblib",compress=3)
    cm=confusion_matrix(ytest,pred,labels=labels)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(reports/"confusion_matrix.csv",index_label="actual")
    pd.DataFrame(classification_report(ytest,pred,labels=labels,output_dict=True,zero_division=0)).T.to_csv(reports/"classification_report.csv",index_label="condition")
    comparison.to_csv(reports/"model_comparison.csv",index=False)
    pd.DataFrame(details).to_csv(reports/"cv_folds.csv",index=False)
    scope.to_csv(reports/"class_scope.csv",index=False)
    manifest=study[["source_row","profile_group","disease"]].copy()
    manifest["split"]="train"; manifest.loc[test_idx,"split"]="test"
    manifest["cv_validation_fold"]=manifest.source_row.map(fold_assignment).astype("Int64")
    manifest.to_csv(reports/"split_manifest.csv",index=False)
    predictions=study.iloc[test_idx][["source_row","profile_group","disease",*SYMPTOMS]].copy()
    predictions["predicted"]=pred; predictions["correct"]=np.asarray(ytest)==pred
    for k in range(3): predictions[f"rank_{k+1}"]=ranked[:,k]
    predictions.to_csv(reports/"test_predictions.csv",index=False)
    pd.DataFrame(stress).to_csv(reports/"unseen_pattern_stress.csv",index=False)
    audit=audit_data(raw,clean,study,scope,source_path); write_json(reports/"data_audit.json",audit)
    evaluation={
        "selected_model":chosen,"selection_metric":"Training-only four-fold grouped macro F1; then balanced accuracy; then simplicity order.",
        "seed":SEED,"train_rows":len(train_idx),"test_rows":len(test_idx),
        "train_groups":int(gtr.nunique()),"test_groups":int(gtest.nunique()),
        "test_fraction":len(test_idx)/len(study),"profile_group_overlap":0,
        "symptom_pattern_overlap":len(set(Xtr.astype(str).agg("".join,axis=1)) & set(Xtest.astype(str).agg("".join,axis=1))),
        "holdout":final,"dummy_holdout":dummy,
        "holdout_accuracy_cluster_bootstrap_95":bootstrap_accuracy(ytest,pred,np.asarray(gtest)),
        "bootstrap_note":"2,000 resamples of holdout raw-profile groups. Conditional on this small split and chosen model; not a clinical guarantee.",
        "labels":labels,"test_missing_labels":sorted(set(labels)-set(ytest)),
        "train_only_unseen_pattern_stress_macro_f1":float(np.mean([r["macro_f1"] for r in stress])),
        "model_refit_on_holdout":False,
        "metric_note":"Macro precision/recall/F1 include all scoped labels (zero for absent labels). Balanced accuracy averages recall over labels present in each evaluation fold. Scores are not clinical probabilities.",
    }
    write_json(reports/"evaluation.json",evaluation)
    metadata={"project":"MediCare Lab","version":"1.0.0","features":list(SYMPTOMS),"labels":labels,
        "selected_model":chosen,"source_sha256":sha256_file(source_path),"model_sha256":sha256_file(models/"model.joblib"),
        "evaluation_sha256":sha256_file(reports/"evaluation.json"),
        "versions":{pkg:importlib.metadata.version(pkg) for pkg in ("numpy","pandas","scipy","scikit-learn","joblib","streamlit")},
        "python":platform.python_version(),"training_source_rows":study.iloc[train_idx].source_row.tolist(),
        "train_rows":len(train_idx),"clinical_use":False,"calibrated":False,
        "model_note":"Exact holdout-evaluated artifact. No post-test refit. All predictions require a visible dataset-limitation notice."}
    write_json(models/"metadata.json",metadata)
    make_plots(reports,comparison,cm,labels,scope)
    return {"audit":audit,"evaluation":evaluation,"comparison":comparison,"model":selected,"scope":scope}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir",type=Path,default=None,help="Optional separate reproduction destination")
    args=parser.parse_args()
    result=run_experiment(args.project_root,args.output_dir)
    print(result["comparison"][["model","cv_macro_f1_mean","cv_accuracy_mean"]].to_string(index=False))
    print(result["evaluation"])


if __name__ == "__main__":
    main()

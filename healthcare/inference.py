"""The same feature contract for the UI, notebook, and tests."""

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import SYMPTOMS, prepare_data, binary_features, sha256_file, symptoms_to_frame


@dataclass
class Bundle:
    model: object
    metadata: dict
    evaluation: dict
    audit: dict
    reference: dict
    clean: pd.DataFrame
    training: pd.DataFrame
    scope: pd.DataFrame


def load_bundle(root: Path) -> Bundle:
    root=Path(root)
    meta=json.loads((root/"models/metadata.json").read_text())
    if meta["features"] != list(SYMPTOMS): raise ValueError("Saved symptom schema mismatch")
    if sklearn.__version__ != meta["versions"]["scikit-learn"]:
        raise ValueError("The saved model needs the scikit-learn version pinned in requirements.txt")
    for path,key in [(root/"models/model.joblib","model_sha256"),(root/"data/raw/Cleaned_Dataset.csv","source_sha256"),(root/"reports/evaluation.json","evaluation_sha256")]:
        if sha256_file(path) != meta[key]: raise ValueError(f"Integrity check failed for {path.name}; restore the package or retrain.")
    # Only the locally generated, checksum-verified artifact is loaded. Never accept uploaded models.
    model=joblib.load(root/"models/model.joblib")
    if list(model.classes_) != meta["labels"]: raise ValueError("Model label metadata mismatch")
    if list(model.feature_names_in_) != list(SYMPTOMS): raise ValueError("Model feature ordering mismatch")
    raw,clean,study,scope=prepare_data(root/"data/raw/Cleaned_Dataset.csv")
    reference=json.loads((root/"data/reference/display_reference.json").read_text(encoding="utf-8"))
    if reference["source_sha256"] != sha256_file(root/"data/reference/medicine_database.source.json"):
        raise ValueError("Reference source changed; rebuild the display subset")
    training=study.loc[study.source_row.isin(meta["training_source_rows"])].copy()
    if len(training)!=meta["train_rows"]: raise ValueError("Training row metadata mismatch")
    return Bundle(model,meta,json.loads((root/"reports/evaluation.json").read_text()),
        json.loads((root/"reports/data_audit.json").read_text()),reference["records"],clean,training,scope)


def predict(bundle: Bundle, selected) -> dict:
    frame=symptoms_to_frame(selected)
    model=bundle.model
    scores=model.predict_proba(frame)[0] if hasattr(model,"predict_proba") else model.decision_function(frame)[0]
    order=np.argsort(-scores,kind="stable")
    prediction=str(model.predict(frame)[0])
    exact=(binary_features(bundle.training).to_numpy()==frame.to_numpy()[0]).all(axis=1)
    matches=bundle.training.loc[exact]
    return {"predicted_condition":prediction,"ranked_conditions":[str(model.classes_[i]) for i in order[:3]],
        "selected_symptoms":[s for s in SYMPTOMS if s in selected],
        "matching_training_rows":len(matches),"matching_training_labels":int(matches.disease.nunique()),
        "training_pattern_seen":bool(len(matches)),
        "clinical_assessment":"Insufficient evidence for a reliable medical conclusion.",
        "educational_only":True}


def condition_info(bundle: Bundle, condition: str) -> dict:
    if condition not in set(bundle.clean.disease): raise ValueError("Unknown supplied condition label")
    examples=bundle.clean.loc[bundle.clean.disease==condition]
    included=condition in bundle.metadata["labels"]
    return {"condition":condition,"source_profiles":len(examples),"included_in_model":included,
        "symptom_counts":{s:int((examples[s]=="Yes").sum()) for s in SYMPTOMS},
        "reference":bundle.reference.get(condition),
        "description":"No disease description was supplied in the available reference dataset."}

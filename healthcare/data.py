"""Explicit feature contract, source validation, and deterministic deduplication."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

SYMPTOMS = ("fever", "cough", "fatigue", "difficulty_breathing")
SYMPTOM_LABELS = {
    "fever": "Fever", "cough": "Cough", "fatigue": "Fatigue",
    "difficulty_breathing": "Difficulty breathing",
}
PROFILE_FIELDS = (*SYMPTOMS, "age", "gender", "blood_pressure", "cholesterol_level")
EXCLUDED_FIELDS = {
    "risk_level": "A deterministic disease-level mapping; unavailable before the prediction.",
    "outcome_variable": "An outcome with undocumented timing; not known at symptom entry.",
    "age_scaled": "Precomputed using undocumented fit data; duplicates age.",
    "bp_scaled": "Precomputed using undocumented fit data; duplicates blood_pressure.",
    "chol_scaled": "Precomputed using undocumented fit data; duplicates cholesterol_level.",
    "age": "Not entered by the symptom-only interface; used only for duplicate grouping.",
    "gender": "Not entered by the symptom-only interface; used only for duplicate grouping.",
    "blood_pressure": "Not entered by the symptom-only interface; used only for duplicate grouping.",
    "cholesterol_level": "Not entered by the symptom-only interface; used only for duplicate grouping.",
}
MIN_CLASS_PROFILES = 5
SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(PROFILE_FIELDS) | {"disease", "risk_level", "outcome_variable", "age_scaled", "bp_scaled", "chol_scaled"}
    if set(df.columns) != required:
        raise ValueError(f"Source schema mismatch. Expected {sorted(required)}; got {list(df.columns)}")
    if df.empty or df.isna().any().any():
        raise ValueError("The supplied source must contain rows and no missing values.")
    for col in (*SYMPTOMS, "disease", "gender", "risk_level", "outcome_variable"):
        df[col] = df[col].astype(str).str.strip()
    for col in SYMPTOMS:
        if not set(df[col]).issubset({"Yes", "No"}):
            raise ValueError(f"Invalid symptom values in {col}")
    if not df["disease"].str.len().gt(0).all():
        raise ValueError("Empty disease label")
    if not df["age"].between(0, 120).all():
        raise ValueError("Age outside the supported source contract")
    if not set(df["gender"]).issubset({"female", "male"}):
        raise ValueError("Unexpected source gender category")
    for col in ("blood_pressure", "cholesterol_level"):
        if not set(df[col]).issubset({0, 1, 2}):
            raise ValueError(f"Unexpected source coding for {col}")
    df.insert(0, "source_row", np.arange(len(df)))
    return df


def profile_hashes(df: pd.DataFrame) -> pd.Series:
    # A proxy for repeated records, NOT a patient identifier. The target is excluded.
    return df[list(PROFILE_FIELDS)].apply(
        lambda row: hashlib.sha256(json.dumps(row.tolist(), separators=(",", ":")).encode()).hexdigest(), axis=1
    )


def binary_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df[list(SYMPTOMS)].apply(lambda col: col.map({"Yes": 1, "No": 0})).astype("int8")
    if not np.isin(result.to_numpy(), [0, 1]).all():
        raise ValueError("Features must be binary")
    return result


def prepare_data(path: Path):
    raw = read_source(path)
    # Predefined eligibility and exact-record cleaning; no learned preprocessing.
    clean = raw.drop_duplicates(subset=[*PROFILE_FIELDS, "disease"]).copy().reset_index(drop=True)
    clean["profile_group"] = profile_hashes(clean)
    support = clean["disease"].value_counts().sort_index()
    scope = support.rename("unique_profiles").rename_axis("condition").reset_index()
    scope["included"] = scope["unique_profiles"] >= MIN_CLASS_PROFILES
    eligible = sorted(scope.loc[scope["included"], "condition"])
    study = clean.loc[clean["disease"].isin(eligible)].copy().reset_index(drop=True)
    if len(eligible) < 2 or len(study) < 30:
        raise ValueError("Insufficient data for the fixed evaluation protocol")
    return raw, clean, study, scope


def symptoms_to_frame(selected) -> pd.DataFrame:
    if isinstance(selected, (str, bytes)):
        raise ValueError("Supply a collection of symptom keys")
    selected = list(selected)
    unknown = set(selected) - set(SYMPTOMS)
    if unknown:
        raise ValueError(f"Unsupported symptoms: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate symptom selections")
    if not selected:
        raise ValueError("Select at least one symptom; an empty input is not a health assessment.")
    return pd.DataFrame([[int(s in selected) for s in SYMPTOMS]], columns=SYMPTOMS, dtype="int8")

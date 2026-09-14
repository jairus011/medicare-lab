"""Tests for leakage boundaries, saved-artifact parity, and source fidelity."""

import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from healthcare.data import SYMPTOMS, binary_features, prepare_data, symptoms_to_frame
from healthcare.inference import load_bundle, predict, condition_info

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT)


def test_raw_source_and_class_scope():
    raw,clean,study,scope=prepare_data(ROOT/"data/raw/Cleaned_Dataset.csv")
    assert (len(raw),len(clean),len(study),raw.disease.nunique(),int(scope.included.sum()))==(349,300,195,116,28)
    assert not clean.duplicated(["profile_group","disease"]).any()
    assert "Other" not in set(study.disease)
    assert len(scope.loc[~scope.included])==88
    assert set(study.disease)==set(scope.loc[scope.included,"condition"])


def test_holdout_and_cv_groups_are_disjoint(bundle):
    m=pd.read_csv(ROOT/"reports/split_manifest.csv")
    assert not m.source_row.duplicated().any()
    assert m.groupby("profile_group").split.nunique().max()==1
    train=m.loc[m.split=="train"]; test=m.loc[m.split=="test"]
    assert train.groupby("profile_group").cv_validation_fold.nunique().max()==1
    assert train.cv_validation_fold.notna().all()
    assert test.cv_validation_fold.isna().all()
    assert set(train.source_row)==set(bundle.metadata["training_source_rows"])
    assert len(train)==157 and len(test)==38
    assert set(train.disease)==set(test.disease)==set(bundle.metadata["labels"])


def test_feature_contract_is_symptom_only(bundle):
    assert list(bundle.model.feature_names_in_)==list(SYMPTOMS)
    assert bundle.model.n_features_in_==4
    assert list(symptoms_to_frame(["cough","fever"]).columns)==list(SYMPTOMS)
    assert symptoms_to_frame(["cough","fever"]).values.tolist()==[[1,1,0,0]]
    assert np.array_equal(bundle.model.predict(symptoms_to_frame(["fever","cough"])),bundle.model.predict(symptoms_to_frame(["cough","fever"])))


@pytest.mark.parametrize("bad",[[],["unknown"],["cough","cough"],"cough",["high_fever"]])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError): symptoms_to_frame(bad)


@pytest.mark.parametrize("bits",list(itertools.product([0,1],repeat=4))[1:])
def test_every_nonempty_pattern(bundle,bits):
    selected=[s for s,b in zip(SYMPTOMS,bits) if b]
    result=predict(bundle,selected)
    assert result["predicted_condition"] in bundle.metadata["labels"]
    assert result["ranked_conditions"][0]==result["predicted_condition"]
    assert result["educational_only"] is True
    assert "confidence" not in result


def test_holdout_metrics_and_saved_model_agree(bundle):
    test=pd.read_csv(ROOT/"reports/test_predictions.csv")
    reloaded=bundle.model.predict(binary_features(test))
    assert reloaded.tolist()==test.predicted.tolist()
    assert np.isclose(accuracy_score(test.disease,reloaded),bundle.evaluation["holdout"]["accuracy"])
    assert np.isclose(f1_score(test.disease,reloaded,labels=bundle.metadata["labels"],average="macro",zero_division=0),bundle.evaluation["holdout"]["macro_f1"])
    assert int((test.disease==reloaded).sum())==2
    assert not bundle.evaluation["model_refit_on_holdout"]


def test_all_displayed_references_are_source_grounded(bundle):
    raw=json.loads((ROOT/"data/reference/medicine_database.source.json").read_text())
    for condition,ref in bundle.reference.items():
        assert ref["description"] is None
        assert not ref["verified_medically"]
        if ref["withheld"]:
            assert condition=="Stroke"
            assert not ref["medicine_names"] and not ref["foods_to_eat"] and not ref["foods_to_avoid"]
            assert ref["source_reported_course"] is None
            continue
        for item in ref["medicine_names"]:
            text=raw[condition][item["source_field"]][item["source_index"]]
            assert item["text"] in text
            assert hashlib.sha256(text.encode()).hexdigest()==item["source_text_sha256"]
            assert not any(ch.isdigit() for ch in item["text"])
        for field in ("foods_to_eat","foods_to_avoid"):
            for item in ref[field]: assert item["text"]==raw[condition][field][item["source_index"]]
        assert ref["source_reported_course"]["text"]==raw[condition]["recovery_time"]


def test_missing_reference_is_explicit(bundle):
    assert condition_info(bundle,"Osteoporosis")["reference"] is None
    assert condition_info(bundle,"Acne")["included_in_model"] is False
    with pytest.raises(ValueError): condition_info(bundle,"Invented condition")


def test_corrupt_or_missing_model_is_rejected(tmp_path):
    for folder in ("models","data","reports"):
        shutil.copytree(ROOT/folder,tmp_path/folder)
    (tmp_path/"models/model.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError,match="Integrity check"):
        load_bundle(tmp_path)
    (tmp_path/"models/model.joblib").unlink()
    with pytest.raises(FileNotFoundError): load_bundle(tmp_path)

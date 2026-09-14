"""Rebuild the README, report, and final notebook from the recorded experiment."""

import argparse
import json
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

import nbformat as nbf
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def md_table(frame):
    lines=["| "+" | ".join(map(str,frame.columns))+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    return "\n".join(lines+["| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None)])


def source_audit():
    return """# Source audit

The final project consolidates two distinct notebook workflows and the supplied
MediCare archive. No absent dataset was reconstructed from truncated notebook
output, downloaded from an unrelated project, or replaced with synthetic records.

## Materials examined

| Source | Finding | Final decision |
| --- | --- | --- |
| `medicare.zip`, `medicare(1).zip` | Byte-identical archives; CSV, Flask app, three HTML templates, two model/encoder artifacts, and a medicine dictionary. The archive's data folder is empty. | Retain the complete CSV and source dictionary; rebuild the training and interface code. |
| `Medicine_Recommendation_System.ipynb`, `Medicine_Recommendation_System (1).ipynb`, `Medicine_Recommendation_System (1)(1).ipynb` | Three byte-identical copies, each 26 cells. Stored outputs describe 4,920 rows and 132 symptom features. | Retain the symptom-to-condition workflow and classifier comparison idea, not the unsupported 100% score. |
| `Personalized_Medicine_Recommending_System (1).ipynb` | 37 cells; a drug-to-drug text similarity demo on a reported 9,720-row dataset. This is a different task. | Audit and exclude from disease prediction and prescribing. |

The original three earlier notebooks are the two `Medicine_Recommendation_System`
variants and the `Personalized_Medicine_Recommending_System` notebook. The current
turn supplied one further identical symptom-notebook copy. SHA-256 checksums and
file sizes are in `source_inventory.json`.

## Confirmed defects in the archive

1. `Cleaned_Dataset.csv` has 349 rows, 14 columns, no missing cells, 116 original
   disease labels, and 49 exact duplicate rows. There are only four binary symptom
   features: at most 16 observable patterns. All 16 map to multiple disease labels.
2. `risk_level` is constant within each of the 116 disease labels. It is a target
   proxy unavailable to a symptom-only user. `outcome_variable` is an outcome with
   undocumented timing. Neither belongs in this prediction contract.
3. All three supplied scaled columns numerically match StandardScaler fitted on
   the complete 349-row dataset (checked with NumPy allclose). That is global
   preprocessing, not training-fold preprocessing. Scaled copies are also redundant.
4. The persisted model's statically inspected schema includes the risk, outcome,
   and scaled fields. The label encoder lists only nine labels, including an
   undocumented `Other` aggregation. Its training source and evaluation are absent.
5. Flask replaces missing scaled values with raw values and missing outcome/risk
   values with zero. It sends numeric symptom values although the source uses
   Yes/No categories. This changes the training input representation.
6. The model chooser changes a response string but still calls the same loaded
   model. A separate hardcoded risk rule has no demonstrated validation. The HTML
   claims 95%+ accuracy and clinical use without supplied evidence.
7. The medicine dictionary and the app's second hardcoded dictionary disagree.
   Some entries contain unverified prescription doses, region-specific emergency
   numbers, and contradictory emergency medication directions. The final app uses
   one traceable source, omits all doses, schedules, advice/triage instructions,
   and withholds Stroke recommendation fields entirely.
8. The contact template includes unrelated contact details and a success message;
   the final project removes this copied contact flow and all diagnostic marketing.

## Symptom notebook audit

- Missing: `Training.csv`, `symtoms_df.csv`, `precautions_df.csv`, `workout_df.csv`,
  `description.csv`, `medications.csv`, and `diets.csv`. The stored outputs do not
  supply those full files. The notebook cannot run from the uploads alone.
- Cell 9 makes one random 70/30 split without stratification or duplicate/group
  checks. Cell 11 compares five classifiers on that same test set; cells 13-17
  reuse it after choosing SVC. No untouched final evaluation is supplied.
- All five displayed accuracies are 1.0. These are historical outputs, not newly
  verified results. Duplicate leakage is a risk, not a confirmed fact for the
  unavailable Training.csv. Its provenance and independence cannot be established.
- Fitting LabelEncoder on the list of labels is not itself predictor leakage;
  label-name enumeration is distinct from learning predictor transformations.
- Cell 21 uses a hardcoded symptom order and label map. The description helper
  reads a global `predicted_disease` instead of its argument. Unknown symptoms
  raise a KeyError; an empty feature vector can still produce a condition label.
- The source saves the SVC alone and relies on separate hardcoded mappings. The
  new model stores its fitted pipeline and validates feature/class metadata.

## Drug similarity notebook audit

- `medicine.csv` is missing. Stored output describes 9,720 medicine catalogue rows;
  this is not a symptom-labelled training dataset.
- CountVectorizer on description plus reason and cosine similarity retrieve text
  neighbors. There is no independent relevance test, prescribing-safety evaluation,
  or evidence that similar text means interchangeable medication.
- Including `Reason` as text is valid catalogue retrieval metadata; it would be
  target leakage only if this were repurposed to predict Reason. No such supervised
  accuracy is claimed here.
- A dense 9,720 x 9,720 similarity matrix has 94,478,400 entries (about 756 MB at
  float64), and is calculated more than once. Query-time sparse similarity would
  be preferable if this distinct retrieval task were resumed.
- Exact-name lookup crashes on missing names. Slicing off the first ranked row
  assumes the self-match comes first; duplicate names/tied similarities can return
  the original drug. The stored example does return its query drug among results.

## What is preserved and what is missing

Preserved: original CSV bytes, original medicine dictionary contents as inert JSON,
the symptom-selection concept, and comparison of standard classifiers. Added:
explicit scope, leakage controls, reproducible artifacts, tests, source fidelity,
and an honest UI. Original model pickles and duplicate notebooks are not shipped
as runnable dependencies. The original dictionary was converted with a restricted
unpickler that forbids global/class loading and persistent references.

No clinical study, source publisher, collection method, patient identifier,
collection date, license, disease descriptions, or verified clinical references
were supplied for the archive data. Do not assert these are real patient records
or that the original authors licensed them for public redistribution. Similar
labels and truncated names remain unchanged; no guessed medical aliases are merged.
"""


def build_text(ev,au,comparison):
    metrics=pd.DataFrame([
        ["Accuracy",f"{ev['holdout']['accuracy']:.2%}",f"{ev['dummy_holdout']['accuracy']:.2%}"],
        ["Macro F1",f"{ev['holdout']['macro_f1']:.4f}",f"{ev['dummy_holdout']['macro_f1']:.4f}"],
        ["Balanced accuracy",f"{ev['holdout']['balanced_accuracy']:.2%}",f"{ev['dummy_holdout']['balanced_accuracy']:.2%}"],
        ["Top-3 accuracy",f"{ev['holdout']['top3_accuracy']:.2%}",f"{ev['dummy_holdout']['top3_accuracy']:.2%}"],
    ],columns=["Holdout metric","Selected random forest","Dummy prior"])
    ci=ev["holdout_accuracy_cluster_bootstrap_95"]
    readme=f"""# MediCare Lab

An educational symptom-to-condition ML prototype, rebuilt from the supplied
Personalized Healthcare & Medicine Recommendation System materials.

**Not medical diagnosis, triage, or a prescribing tool.** The selected model
correctly predicts **2 of 38 held-out records ({ev['holdout']['accuracy']:.2%})**.
It is not clinically useful. Completing the software does not validate its
medical predictions. No fabricated medical descriptions or recommendations are used.

![Streamlit symptom explorer](screenshots/symptom_explorer.png)

## Run on Windows (PowerShell)

Use Python 3.12. Extract the project ZIP and open a terminal in `healthcare_final`.
Using the virtual environment's Python directly avoids activation-policy problems.

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. The bundled model is ready to use;
training is not required to launch. The server binds to localhost by default.

## Run on Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Reproduce the experiment and run tests

With the environment active (or substitute its full Python path):

```bash
python -m pip install -r requirements-dev.txt
python -m healthcare.train --output-dir reproduction
python -m pytest
```

`requirements-lock.txt` also pins the resolved transitive dependencies for Python
3.12, with platform markers for Windows/macOS/Linux. Use it when you want a full
development-environment lock: `python -m pip install -r requirements-lock.txt`.
The recorded runtime and test results were produced on Linux; Windows launch
instructions are supplied, but no Windows machine was available for testing.

`reproduction/models` and `reproduction/reports` are independent outputs; the
shipped holdout and model remain available for comparison. `python -m
healthcare.train` without `--output-dir` deliberately rebuilds the local models
and reports. Run it only when that overwrite is intended. The final notebook
reruns the experiment into a temporary directory and asserts identical predictions,
scores, selected model, and split manifest against the shipped results.

Rebuild source display entries with `python scripts/prepare_references.py`.
Rebuild the human-readable deliverables with `python scripts/build_deliverables.py`.
To execute a freshly rebuilt notebook: `python scripts/execute_notebook.py --in-process`.
This executes real cells through IPython and captures their actual rich outputs.
That mode was verified here; this container restricts standalone kernel sockets.
Normal Jupyter execution is also available by omitting `--in-process` on a system
that supports kernel networking. The notebook can be opened normally in VS Code/Jupyter.
Optional screenshots: `python -m playwright install chromium`, then run
`python scripts/browser_smoke.py --start-server`. The test starts and stops the
local Streamlit server automatically. No website is published by these commands.

## What the app does

- **Symptom explorer:** select fever, cough, fatigue, and/or difficulty breathing;
  explicitly confirm that unselected symptoms are absent; see a predicted label,
  two other ranked labels, and ambiguity among matching training records.
- **Condition reference:** browse all 116 exact source labels, including labels
  outside the classifier. Ten labels have source reference entries; nine can show
  a limited, unverified display subset. Missing data stays visibly missing.
- **Model & data:** inspect CV comparisons, holdout results, class coverage,
  exclusions, and the confusion matrix. No probability-of-disease claim is made.

The app has no login or user-history database. It does not log selected symptoms
or contact external APIs. Streamlit usage telemetry is disabled. Selections exist
only in the current session. No real personal information is needed for a demo.

## Dataset and scope

The only complete labelled table supplied is `Cleaned_Dataset.csv`: 349 rows,
116 disease labels, and four symptom features. Removing 49 exact duplicate
profile/label records leaves 300. A **predefined five-profile minimum** retains
195 records across 28 labels for this small experiment; 105 records across 88
labels are excluded. No `Other` class or medical alias guesses are introduced.

This eligibility check uses the source label counts to define the study before
splitting. It is a conditional benchmark, not performance across all 116 labels.
Out-of-scope diseases cannot be detected, excluded, or ruled out by the model.

The notebook's 4,920-row `Training.csv`, its six reference CSVs, and the separate
drug-similarity notebook's `medicine.csv` are unavailable. The three symptom
notebook uploads are identical; the drug-similarity notebook is a separate task.
See [the complete source audit](reports/source_audit.md).

## Evaluation design

1. Preserve the original CSV and checksum. Deduplicate identical raw profiles
   with the same label; retain conflicting labels without inventing a resolution.
2. Group records by all eight raw pre-outcome profile fields, without the target.
   This keeps identical full profiles together. It is a duplicate-group proxy,
   not proof of patient independence; patient/site/time identifiers are absent.
3. Reserve the first fixed five-fold StratifiedGroupKFold partition (seed 42):
   **157 training records / 38 test records**, with all 28 labels in both sets.
4. Compare seven fixed candidate classifiers in four grouped folds of training
   data. Choose by macro F1, then balanced accuracy, then predefined simplicity.
   No hyperparameter search or model choice uses the test results.
5. Fit the winner on the 157 training records; evaluate it and the dummy baseline
   on the held-out set. Save that exact evaluated model; do not refit on test data.

Only the four raw binary symptoms enter the pipeline. No scaling is necessary.
The conversion step is stateless and serialized with the classifier. Outcome,
disease-derived risk, globally scaled copies, and non-UI context fields are
excluded. The original scaled columns match transforms fitted on all source rows.
The full-profile train/test group overlap is zero. Thirteen four-symptom patterns
are shared by different profiles across the split; that is reported explicitly.
An additional training-only stress test holds out entire symptom patterns and has
macro F1 **{ev['train_only_unseen_pattern_stress_macro_f1']:.4f}**. It is not used for selection.

The use of consistent preprocessing and training-only fitting follows
[scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
Group separation uses [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

## Measured results

{md_table(metrics)}

The winner is **{ev['selected_model']}**, selected by CV macro F1
**{comparison.iloc[0].cv_macro_f1_mean:.4f}**. It underperforms the dummy in test
accuracy and top-3 accuracy, although its macro F1 is higher. It remains the saved
CV-selected artifact; the test set was not used to switch models afterward.

The 95% group-bootstrap interval for test accuracy is **{ci[0]:.1%} to {ci[1]:.1%}**
(2,000 resamples of only 33 holdout profile groups). This is a very uncertain,
conditional estimate, not a clinical confidence statement. Precision, recall,
F1 and support for each label, error rows, CV folds, and the split manifest are
included in `reports/`. AUROC and calibrated probabilities are not advertised:
this small multiclass split cannot support a meaningful clinical calibration claim.

## Source-grounded reference policy

`medicine_database.source.json` is an inert, exact-content conversion of the
provided dictionary **for audit, not clinical use**. `display_reference.json`
contains only manually reviewed exact medicine-name substrings, verbatim diet
entries, and source-reported course text. Every item points back to its source
field/index; medication substrings carry a source-text checksum. Reference data
is never a training feature. All entries are unverified and opt-in in the UI.

The source has no disease-description field. The app says so. It does not use
an LLM or the internet to invent missing text. Dose schedules, advice and triage
instructions, and region-specific numbers are omitted. Stroke recommendation
fields are withheld because the source contains conflicting emergency directions.
No drug similarity results are offered as medication substitutes.

## Project contents

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit application |
| `healthcare/data.py` | Input schema, validation, scope, duplicate grouping |
| `healthcare/train.py` | Reproducible model selection, holdout evaluation and export |
| `healthcare/inference.py` | Artifact checks, shared inference, reference lookups |
| `data/raw/` | Original complete CSV |
| `data/reference/` | Inert source JSON and traceable display subset |
| `models/` | Exact evaluated pipeline and metadata/checksums |
| `notebooks/healthcare_final.ipynb` | Executed final analysis and reproducibility checks |
| `reports/` | Source audit, data audit, split and error tables, metrics, figures, PDF |
| `screenshots/` | Actual browser captures and browser test summary |
| `tests/` | Data, artifact, source fidelity and Streamlit tests |
| `scripts/` | Reference, document, notebook and browser reproduction tools |

## Limitations and next requirements

This is a completed educational software project with an inadequate clinical
dataset. There is no demonstrated personalization, external validation, patient
independence, reliable diagnosis, validated treatment recommendation, fairness
assessment, or clinical outcome evidence. Source provenance and reuse license
are undocumented. No claim is made that the records are real patients.

Before a serious research extension: recover the actual missing datasets and
their licenses; obtain richer, clinically governed symptom data with patient/site/
time identifiers; define intended use and exclusions; have qualified reviewers
validate reference content; then pre-register and run new external evaluation.
Do not tune repeatedly against the included 38-record holdout.

## Troubleshooting

- **Model/data not found:** extract the whole ZIP and launch from its project
  directory. Do not move `app.py` out of the project.
- **Version or integrity mismatch:** install the exact requirements in a fresh
  Python 3.12 environment, restore the shipped files, or explicitly retrain.
- **Port 8501 in use:** add `--server.port 8502` and open that port instead.
- **Notebook kernel missing:** install development requirements and select that
  environment as the notebook kernel in VS Code/Jupyter.

Load only the bundled model or models you trained and trust; joblib/pickle files
can execute code. The app never accepts user-uploaded model files. Checksums
detect accidental changes; they are not a digital signature or an authenticity proof.
"""
    (ROOT/"README.md").write_text(readme,encoding="utf-8")
    (ROOT/"reports/source_audit.md").write_text(source_audit(),encoding="utf-8")
    modelcard=f"""# Model card: MediCare Lab v1.0

- Intended use: educational inspection of a weak symptom classifier and source lookups.
- Prohibited use: diagnosis, triage, prescribing, medication substitution, or clinical decisions.
- Selected model: {ev['selected_model']}; seed 42; fixed training-only grouped CV selection.
- Input: four reviewed binary symptoms; unselected means absent, not unknown.
- Output: one of 28 exact original condition labels. No calibrated clinical probabilities.
- Training: 157 source records in 139 raw-profile groups; test: 38 records in 33 groups.
- Test accuracy: {ev['holdout']['accuracy']:.4f}; macro F1: {ev['holdout']['macro_f1']:.4f}.
- Baseline accuracy: {ev['dummy_holdout']['accuracy']:.4f}; baseline macro F1: {ev['dummy_holdout']['macro_f1']:.4f}.
- 88 source labels excluded by a predefined minimum-support rule; unknown-condition detection is absent.
- All four-symptom patterns are ambiguous in the complete source. No patient IDs were supplied.
- No external clinical validation, calibration, demographic fairness conclusion, or personalized treatment evidence.
- Exact evaluated pipeline is saved; no full-data refit. Reference data does not enter model training.
- Runtime, data checksum and model checksum: `models/metadata.json`.
- Missing descriptions and references are explicit. Unverified reference entries are opt-in; doses are omitted.
- Source content and reuse permissions have not been independently established.
"""
    (ROOT/"reports/model_card.md").write_text(modelcard,encoding="utf-8")
    return readme,metrics


def build_notebook(ev,au):
    cells=[]
    def cell_text(s):
        lines=s.splitlines()
        return lines[0].strip()+"\n"+textwrap.dedent("\n".join(lines[1:])).rstrip()
    def md(s): cells.append(nbf.v4.new_markdown_cell(cell_text(s)))
    def code(s): cells.append(nbf.v4.new_code_cell(cell_text(s)))
    md("""# MediCare Lab - final project
    **Educational ML prototype. Not medical diagnosis or prescribing.**

    This notebook audits the supplied materials, reproduces model selection without
    consulting test scores, evaluates the exact saved pipeline, and checks the app's
    input and source-reference contracts. Actual measured accuracy is low; no claim
    of clinical usefulness is made. Run all cells from top to bottom using Python 3.12
    and `requirements-dev.txt`. No downloads, interactive input, or external datasets
    are required. The shipped artifacts are not overwritten by this notebook.
    """)
    code("""from pathlib import Path
    import json, sys, tempfile
    import numpy as np
    import pandas as pd
    from IPython.display import display, Image, Markdown
    ROOT = Path.cwd().resolve()
    if not (ROOT / 'healthcare').is_dir():
        ROOT = ROOT.parent
    assert (ROOT / 'healthcare').is_dir(), 'Run from the project or notebooks directory'
    sys.path.insert(0, str(ROOT))
    from healthcare.data import prepare_data, binary_features, SYMPTOMS
    from healthcare.train import run_experiment
    from healthcare.inference import load_bundle, predict, condition_info
    bundle = load_bundle(ROOT)
    display(pd.Series(bundle.metadata['versions'], name='tested version'))
    """)
    md("""## 1. Source inventory and notebook audit
    The archive variants are byte-identical. All three symptom-notebook copies
    are byte-identical. A separate personalized-medicine notebook performs text
    similarity, not symptom classification. Its catalogue dataset is absent.

    The symptom notebook's Training.csv and six supporting CSVs are also missing.
    Stored 100% scores cannot be verified. A single test split was reused for model
    comparison; there is no protected final test or duplicate audit. Its label
    encoder fitted on label names is not itself predictor leakage. The complete
    source audit distinguishes confirmed defects from unverified leakage risks.
    """)
    code("""inventory = json.loads((ROOT / 'reports/source_inventory.json').read_text())
    display(pd.DataFrame(inventory['files'])[['name', 'bytes', 'sha256', 'role']])
    display(Markdown((ROOT / 'reports/source_audit.md').read_text()))
    """)
    md("""## 2. Audit the complete available table
    Keep the source bytes unchanged. Remove exact raw-profile/label duplicates;
    keep contradictory labels rather than guessing a correction. Define eligibility
    at five unique profiles per label before creating the split. This limits the
    benchmark to a subset of source conditions and is not open-world diagnosis.
    """)
    code("""raw, clean, study, scope = prepare_data(ROOT / 'data/raw/Cleaned_Dataset.csv')
    display(pd.Series(bundle.audit))
    display(raw.head())
    display(scope.sort_values('unique_profiles', ascending=False).head(35))
    assert len(raw) - len(clean) == 49
    assert len(study) == 195 and study.disease.nunique() == 28
    """)
    code("""from sklearn.preprocessing import StandardScaler
    scaled_pairs = [('age', 'age_scaled'), ('blood_pressure', 'bp_scaled'), ('cholesterol_level', 'chol_scaled')]
    global_scale_check = {a: np.allclose(StandardScaler().fit_transform(raw[[a]]).ravel(), raw[b]) for a, b in scaled_pairs}
    display(pd.Series(global_scale_check, name='matches scaler fitted on ALL source rows'))
    display(raw.groupby('disease').risk_level.nunique().value_counts().rename('disease labels'))
    ambiguity = raw.groupby(list(SYMPTOMS)).disease.nunique().rename('distinct labels').reset_index()
    display(ambiguity)
    assert ambiguity['distinct labels'].gt(1).all()
    """)
    md("""**Interpretation.** The supplied scaled columns match full-dataset scaling.
    They and the disease-derived risk field are excluded. The outcome's timing is
    undocumented, so it is excluded too. Age, gender, blood pressure and cholesterol
    are used only to group repeated profiles, never as inputs to this symptom-only
    interface. Four binary inputs cannot uniquely encode these labels. The computed
    majority-per-pattern ceiling is a descriptive in-sample bound, not validation.
    """)
    code("""display(Image(filename=str(ROOT / 'reports/figures/class_support.png')))
    """)
    md("""## 3. Fixed evaluation protocol
    Use the first five-fold StratifiedGroupKFold partition with seed 42 as the
    holdout. Compare seven fixed candidate classifiers in four grouped CV folds
    within training. Select by macro F1, then balanced accuracy, then predefined
    simplicity order. Evaluate only the selected model and the dummy baseline on
    the holdout. Save the exact evaluated fit; do not train on the holdout afterward.

    Full raw-profile groups cannot cross train/test or CV partitions. They are a
    proxy, not patient identifiers. Different profiles can share symptoms across
    the split. A supplementary training-only GroupKFold challenge holds out whole
    symptom patterns and is never used to choose a model.
    """)
    code("""manifest = pd.read_csv(ROOT / 'reports/split_manifest.csv')
    display(pd.crosstab(manifest.disease, manifest.split))
    assert manifest.groupby('profile_group').split.nunique().max() == 1
    assert manifest.query("split == 'train'").groupby('profile_group').cv_validation_fold.nunique().max() == 1
    display(pd.Series(manifest.split.value_counts(), name='split rows'))
    """)
    md("""## 4. Reproduce training and compare classifiers
    The following reruns the fixed experiment in a temporary directory, then
    asserts parity with the shipped artifacts. These checks verify reproducibility;
    they do not search for a better test score or change the model after evaluation.
    """)
    code("""reproduction_dir = tempfile.TemporaryDirectory(prefix='medicare_reproduction_')
    reproduced = run_experiment(ROOT, Path(reproduction_dir.name))
    comparison = reproduced['comparison']
    display(comparison[['model', 'cv_macro_f1_mean', 'cv_macro_f1_std', 'cv_accuracy_mean', 'cv_balanced_accuracy_mean']])
    assert reproduced['evaluation'] == bundle.evaluation
    for filename in ['split_manifest.csv', 'test_predictions.csv', 'model_comparison.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(ROOT / 'reports' / filename), pd.read_csv(Path(reproduction_dir.name) / 'reports' / filename))
    print('Verified identical selection, scores, test predictions, and group partitions.')
    display(Image(filename=str(ROOT / 'reports/figures/model_comparison.png')))
    """)
    md("""## 5. Final held-out evaluation and error analysis
    Accuracy counts correct labels. Macro F1 weights every configured condition
    equally; macro precision/recall/F1 include all 28 labels with zero division
    handled explicitly. Balanced accuracy averages recall across labels present
    in each fold. Top-3 is a ranking metric, not a clinical differential diagnosis.
    Confidence intervals are conditional on this small, fixed dataset and split.
    """)
    code("""ev = bundle.evaluation
    display(pd.DataFrame({'selected model': ev['holdout'], 'dummy prior': ev['dummy_holdout']}))
    print('95% profile-group bootstrap accuracy interval:', ev['holdout_accuracy_cluster_bootstrap_95'])
    display(pd.read_csv(ROOT / 'reports/classification_report.csv'))
    errors = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    display(errors.loc[~errors.correct, ['source_row', 'disease', 'predicted', *SYMPTOMS]].head(15))
    display(Image(filename=str(ROOT / 'reports/figures/confusion_matrix.png')))
    """)
    md(f"""**Result.** Random forest was chosen by training CV macro F1
    ({ev['selection_metric']}). On the untouched holdout, accuracy is
    {ev['holdout']['accuracy']:.2%} and macro F1 is {ev['holdout']['macro_f1']:.4f}.
    It is worse than the dummy baseline in accuracy and top-3 accuracy. The dataset
    is inadequate for reliable diagnosis; no attempt is made to conceal or optimize
    away this negative result by switching models on the test data.
    """)
    code("""stress = pd.read_csv(ROOT / 'reports/unseen_pattern_stress.csv')
    display(stress)
    print('Training-only unseen-pattern mean macro F1:', ev['train_only_unseen_pattern_stress_macro_f1'])
    """)
    md("""## 6. Saved model and application inference
    The application calls the same `predict` function below. Symptom order is
    fixed by a saved schema, unselected symptoms mean absent after confirmation,
    and empty/unknown inputs are rejected. No extra clinical inputs are invented.
    """)
    code("""example = predict(bundle, ['cough', 'difficulty_breathing'])
    display(example)
    test_rows = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    assert bundle.model.predict(binary_features(test_rows)).tolist() == test_rows.predicted.tolist()
    display(condition_info(bundle, example['predicted_condition']))
    """)
    md("""## 7. Source-grounded information and recommendations
    The archive's medicine dictionary supplies ten reference records but no disease
    descriptions. Exact medicine-name substrings and verbatim food/course entries
    are exposed only as unverified study material. Missing entries are explicit.
    No dosage, medication substitution, emergency medication instructions, or
    generated medical text is displayed. Stroke recommendation fields are withheld
    because of internal source contradictions. Clinical verification is absent.
    """)
    code("""reference = bundle.reference['Asthma']
    display({k: reference[k] for k in ['source', 'verified_medically', 'description_status', 'medicine_names']})
    assert all(not any(ch.isdigit() for ch in item['text']) for item in reference['medicine_names'])
    assert bundle.reference['Stroke']['withheld']
    assert condition_info(bundle, 'Osteoporosis')['reference'] is None
    assert not condition_info(bundle, 'Acne')['included_in_model']
    """)
    md("""## 8. Validation evidence
    Functional tests execute the real Streamlit app using its AppTest API. They
    cover all 15 non-empty input patterns, empty-input/confirmation handling,
    stale-result removal, present/missing/withheld references, and the evaluation
    page. Additional tests check group separation, source fidelity and artifact
    integrity. See the recorded test files; rerun with `python -m pytest`.
    """)
    code("""import xml.etree.ElementTree as ET
    results = ET.parse(ROOT / 'reports/test_results.xml').getroot()
    display([s.attrib for s in results.iter('testsuite')])
    browser_path = ROOT / 'screenshots/browser_checks.json'
    if browser_path.exists():
        display(json.loads(browser_path.read_text()))
    reproduction_dir.cleanup()
    """)
    md("""## Final conclusion
    The software is complete and reproducible. The supplied dataset does **not**
    establish clinical capability. Recover the missing original datasets and their
    provenance, obtain richer governed data with patient/site/time grouping,
    validate all reference information with qualified reviewers, and define a new
    external evaluation before pursuing any health-related real-world use.

    Technical references: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html),
    [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
    [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
    These sources support implementation practices, not medical information.
    """)
    notebook=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3.12 (healthcare)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
    nbf.write(notebook,ROOT/"notebooks/healthcare_final.ipynb")


def build_pdf(ev,au,comparison,metrics):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('ReportBody',str(fontdir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportBold',str(fontdir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('ReportBody',normal='ReportBody',bold='ReportBold',italic='ReportBody',boldItalic='ReportBold')
    navy=colors.HexColor('#163047');teal=colors.HexColor('#087f8c');gray=colors.HexColor('#536b7b')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName='ReportBold',fontSize=25,leading=29,textColor=navy,spaceAfter=13))
    styles.add(ParagraphStyle(name='Kicker',fontName='ReportBold',fontSize=8,leading=11,textColor=teal,spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyCustom',fontName='ReportBody',fontSize=9.0,leading=13.2,textColor=navy,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='ReportBody',fontSize=7.7,leading=10.6,textColor=gray,spaceAfter=7))
    styles.add(ParagraphStyle(name='HeadingCustom',fontName='ReportBold',fontSize=12.5,leading=16,textColor=teal,spaceBefore=13,spaceAfter=8))
    story=[]
    def para(s,style='BodyCustom'): story.append(Paragraph(s,styles[style]))
    def table(rows,widths):
        body=[[Paragraph(escape(str(c)),styles['SmallCustom']) for c in row] for row in rows]
        t=Table(body,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f3f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dbe4ea')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),8)]))
        story.append(t)
    para('JOB 1 / HEALTHCARE / FINAL PROJECT REPORT','Kicker')
    para('MediCare Lab','TitleCustom')
    para('A reproducible symptom-classification project with transparent limits.','HeadingCustom')
    para('<b>Software complete. Clinical capability not demonstrated.</b> The selected random forest correctly predicts 2 of 38 held-out records (5.3%). Its accuracy is below the majority baseline. The application is explicitly an educational ML prototype, not medical diagnosis, triage or a prescribing tool.')
    table([['Source records','Model study','Final evaluation'],['349 rows / 116 labels','195 records / 28 labels','157 train / 38 holdout'],['49 duplicate rows removed','4 binary symptom inputs','0 overlapping profile groups']],[165,165,165])
    para('1. Audit findings','HeadingCustom')
    para('The two archive copies are identical. Three symptom-notebook copies are identical; a fourth notebook is a distinct drug-similarity workflow. All available code, notebook cells, stored outputs, data, templates and artifact schemas were inspected. File hashes are preserved in the source inventory.')
    para('<b>Unavailable inputs.</b> The symptom notebook requires Training.csv and six reference CSVs that were not supplied. The drug-similarity notebook requires medicine.csv, also absent. Their stored outputs cannot reproduce those datasets or validate the old 100% accuracy claims.')
    para('<b>Confirmed leakage and inference defects.</b> The archive includes disease-derived risk and an outcome in the old model schema. Three scaled fields match scaling on the entire source table. The Flask app substitutes raw values for scaled values and zeros for unavailable fields, with mismatched symptom types. The model chooser still calls one model; clinical accuracy claims are unsupported.')
    para('<b>Source-content defects.</b> Two medication dictionaries disagree. Unverified doses and contradictory emergency directions appear in the source. The new app exposes only a traceable subset, removes dose/triage instructions and withholds Stroke recommendation fields. No medical facts were invented.')
    para('2. Foundation and limits','HeadingCustom')
    para('The complete archive CSV is the reproducible foundation. Removing 49 exact profile/label duplicates leaves 300 records. A predefined minimum of five distinct profiles per original label retains 195 records across 28 labels; 105 records across 88 labels are outside scope. No guessed label merges or Other class are introduced.')
    para('All 16 possible symptom patterns map to multiple labels in the complete dataset. There are no supplied patient IDs, dates, sites, clinical references or documented collection provenance. Full-profile grouping reduces duplicate contamination; it does not establish patient independence.','SmallCustom')
    story.append(PageBreak())
    para('EXPERIMENT / SELECTION / HELD-OUT RESULTS','Kicker')
    para('3. Reproducible evaluation','TitleCustom')
    para('Only fever, cough, fatigue and difficulty breathing enter the fitted pipeline. Outcome, risk, pre-scaled fields and non-UI context are excluded. Binary conversion is stateless; no learned scaling or imputation is needed.')
    para('The first five-fold StratifiedGroupKFold split at seed 42 reserves the holdout. Groups hash all eight raw pre-outcome profile fields without the target. Four grouped training folds compare seven fixed classifiers; selection uses macro F1, then balanced accuracy, then predefined simplicity. The test set is evaluated after selection. The exact evaluated fit is saved, without refitting on the holdout.')
    rows=[['Classifier','CV macro F1','CV accuracy']]
    for row in comparison.itertuples(): rows.append([row.model,f'{row.cv_macro_f1_mean:.4f}',f'{row.cv_accuracy_mean:.2%}'])
    table(rows,[255,120,120])
    para('4. Results and interpretation','HeadingCustom')
    table([list(metrics.columns),*metrics.values.tolist()],[165,165,165])
    para('Random forest wins training CV macro F1 (0.0452), but does not beat the dummy baseline in holdout accuracy or top-3 ranking. Its higher macro F1 does not establish medical usefulness. This negative result is retained rather than changing the selected model after inspecting the test set.')
    para('The 95% holdout accuracy interval is 0.0%-13.9%, from 2,000 resamples of 33 raw-profile groups. With only 38 test records across 28 labels, estimates are unstable. Per-class precision/recall/F1/support, error rows, CV fold scores, confusion matrices and exact split assignments are included.')
    para(f'Thirteen symptom patterns occur in both partitions through different raw profiles. An additional training-only stress test holds out entire symptom patterns: mean macro F1 {ev["train_only_unseen_pattern_stress_macro_f1"]:.4f}. It is a separate robustness challenge, not a selection criterion. No calibrated clinical probability or AUROC claim is made.','SmallCustom')
    para('Method references: <link href="https://scikit-learn.org/stable/common_pitfalls.html" color="#087f8c">scikit-learn leakage and preprocessing guidance</link>; <link href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html" color="#087f8c">StratifiedGroupKFold</link>. These support technical practice only.','SmallCustom')
    story.append(PageBreak())
    para('APPLICATION / DELIVERY / NEXT REQUIREMENTS','Kicker')
    para('5. Finished application','TitleCustom')
    para('<b>Symptom explorer.</b> Users select from the four supported symptoms and confirm that unselected symptoms mean absent. The app displays a model label, two other ranked labels, matching-training-record ambiguity, and a prominent educational notice. It rejects empty input and removes stale results after input changes.')
    para('<b>Condition reference.</b> All 116 exact source labels can be browsed. Ten have supplied reference records; nine expose a limited, unverified subset behind an explicit study checkbox. Each name or excerpt points to the source field/index. Descriptions are unavailable and are labelled missing. Drug names are not treatment suggestions; no doses or substitute medicines are supplied.')
    para('<b>Model and data.</b> The interface exposes actual metrics, baseline performance, exclusions, class coverage and figures. It collects no names or contact details, stores no symptom history and uses no external medical API. Usage telemetry is disabled and local launch binds to localhost.')
    para('6. Testing and deliverables','HeadingCustom')
    para('31 automated tests cover data scope, split/CV isolation, feature ordering, invalid inputs, all 15 non-empty symptom combinations, model reload parity, metric recomputation, missing/corrupt model files, exact source links, missing/withheld references and real Streamlit execution. The executed notebook independently reproduces selection, scores, predictions and splits.')
    browser_path=ROOT/'screenshots/browser_checks.json'
    if browser_path.exists():
        browser=json.loads(browser_path.read_text())
        para('The live Streamlit server was additionally exercised in Chromium. Desktop and mobile layouts, prediction, reference disclosure and model navigation were checked. Actual screenshots and browser-check results are included.')
    para('Included: application and reusable Python modules; exact requirements; original CSV and source-reference JSON; saved model and checksums; executed final notebook; README; source audit; model card; full evaluation tables and figures; automated tests; reproduction scripts; this report and browser screenshots.')
    para('7. Run and next steps','HeadingCustom')
    para('Use Python 3.12 in a virtual environment. Install requirements.txt, then run <b>python -m streamlit run app.py</b>. The README includes Windows PowerShell commands that do not require environment activation. For testing and notebooks, install requirements-dev.txt. Reproduce into a separate folder with <b>python -m healthcare.train --output-dir reproduction</b>.')
    para('Before any clinical research extension: recover the missing datasets and their provenance/reuse terms; obtain richer governed symptom records with patient, site and time identifiers; define intended use; clinically review every reference entry; then perform a new, independently planned external evaluation. Do not repeatedly tune against this holdout.')
    para('No clinical deployment, medical certification, personalization, demographic fairness validation, or reliable diagnosis is claimed. Source reuse permissions remain undocumented.','SmallCustom')
    para('Testing reference: <link href="https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest" color="#087f8c">Streamlit AppTest documentation</link>. Full technical and source audit details are in the accompanying repository.','SmallCustom')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dbe4ea'));canvas.line(50,43,545,43)
        canvas.setFillColor(gray);canvas.setFont('ReportBody',7.0)
        canvas.drawString(50,29,'MEDICARE LAB | Educational prototype - not medical diagnosis')
        canvas.drawRightString(545,29,str(doc.page));canvas.restoreState()
    path=ROOT/'reports/Healthcare_Project_Report.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=58,title='MediCare Lab - Healthcare Project Report',author='MediCare Lab project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-notebook',action='store_true',help='Preserve an already executed notebook')
    args=parser.parse_args()
    ev=json.loads((ROOT/'reports/evaluation.json').read_text())
    au=json.loads((ROOT/'reports/data_audit.json').read_text())
    comparison=pd.read_csv(ROOT/'reports/model_comparison.csv')
    readme,metrics=build_text(ev,au,comparison)
    if not args.keep_notebook: build_notebook(ev,au)
    build_pdf(ev,au,comparison,metrics)
    print('Created README, audit, model card, final notebook and PDF report.')


if __name__=='__main__': main()

# Source audit

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

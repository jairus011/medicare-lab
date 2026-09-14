"""Run from the project directory: python -m streamlit run app.py"""

from pathlib import Path
import html

import pandas as pd
import streamlit as st

from healthcare.data import SYMPTOMS, SYMPTOM_LABELS
from healthcare.inference import load_bundle, predict, condition_info

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="MediCare Lab | Educational ML",page_icon="✚",layout="wide",initial_sidebar_state="auto")

st.markdown("""
<style>
.stApp {background: #f6f8fa;}
.block-container {padding-top:4.5rem; padding-bottom:3rem; max-width:1280px;}
[data-testid="stSidebar"] {background:#163047;}
[data-testid="stSidebar"] * {color:#f4f8fb;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#b7cbd8;}
h1 {font-size:2.65rem !important;letter-spacing:-.055em;line-height:1.12 !important;}
h2 {font-size:1.6rem !important;letter-spacing:-.025em;}
h3 {font-size:1.17rem !important;}
.eyebrow {color:#087f8c;letter-spacing:.16em;font-size:.73rem;font-weight:750;margin-bottom:10px;}
.lead {font-size:1.04rem;color:#567083;line-height:1.6;max-width:780px;margin-bottom:24px;}
.brand {font-size:1.55rem;font-weight:750;letter-spacing:-.04em;margin:8px 0 2px;}
.result {border-left:4px solid #087f8c;background:#e9f5f4;border-radius:5px 16px 16px 5px;padding:22px 25px;margin:12px 0 18px;}
.result-label {font-size:.73rem;letter-spacing:.1em;font-weight:700;color:#55747e;}
.result-title {font-size:1.85rem;font-weight:750;letter-spacing:-.03em;color:#163047;margin:7px 0;}
.result-note {font-size:.92rem;line-height:1.5;color:#3d5d6d;}
.footer-note {font-size:.8rem;color:#6c8090;border-top:1px solid #dbe4eb;padding-top:17px;margin-top:25px;}
[data-testid="stMetric"] {background:white;border:1px solid #dfe7ed;border-radius:13px;padding:16px;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
button[kind="primary"] {border-radius:9px;min-height:43px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
@media (max-width:700px) {.block-container {padding:4.6rem 1.2rem 2rem;}h1 {font-size:2rem !important;}.result-title{font-size:1.4rem;}}
</style>
""",unsafe_allow_html=True)


@st.cache_resource
def resources():
    return load_bundle(ROOT)


try:
    bundle=resources()
except (FileNotFoundError,ValueError,KeyError,ImportError,TypeError) as exc:
    st.error("The project data or model could not be loaded. Restore the supplied files, install requirements.txt, and run `python -m healthcare.train`.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand">✚ MediCare Lab</div>',unsafe_allow_html=True)
    st.caption("Explore symptoms. Understand the limits.")
    st.divider()
    page=st.radio("Workspace",["Symptom explorer","Condition reference","Model & data"],key="navigation")
    st.divider()
    st.caption("EDUCATIONAL ML PROTOTYPE")
    st.write("Not medical diagnosis or a prescribing tool.")
    st.caption("Four symptoms cannot reliably distinguish these conditions. Results are exploratory dataset labels.")
    st.divider()
    st.caption("No login • No symptom history stored")


def show_reference(condition, key):
    info=condition_info(bundle,condition)
    st.subheader("Supporting dataset information")
    scope_text=f"Included in the {len(bundle.metadata['labels'])}-label model" if info['included_in_model'] else "Outside the model scope"
    st.caption(f"{info['source_profiles']} distinct supplied profiles · {scope_text}")
    st.info(info["description"])
    ref=info["reference"]
    if ref is None:
        st.info("No medicine or recommendation record was supplied for this exact condition label.")
        return
    st.caption("Reference source: "+ref["source"])
    if ref["withheld"]:
        st.warning(ref["withheld_reason"])
        return
    st.caption("These entries come from the uploaded project, which supplies no clinical references or validation. They are not personalized advice.")
    if st.checkbox("Show unverified source entries for study",key=f"references_{key}"):
        st.warning("Reference material only. Do not start, stop, substitute, or dose a medicine based on this prototype. Discuss real health concerns with a qualified healthcare professional.")
        left,right=st.columns(2,gap="large")
        with left:
            st.markdown("**Medicine names mentioned in the source**")
            for item in ref["medicine_names"]: st.text("• "+item["text"])
            st.caption("Names only; not treatment recommendations. Source doses and schedules are excluded.")
        with right:
            st.markdown("**Source-reported course**")
            st.text(ref["source_reported_course"]["text"])
            st.caption("Unverified source statement; not a prognosis for the user.")
        with st.expander("Diet entries copied from the source"):
            eat,avoid=st.columns(2)
            with eat:
                st.markdown("**Source field: foods_to_eat**")
                for item in ref["foods_to_eat"]: st.text("• "+item["text"])
            with avoid:
                st.markdown("**Source field: foods_to_avoid**")
                for item in ref["foods_to_avoid"]: st.text("• "+item["text"])
            st.caption("These lists may be incomplete or inaccurate and are not dietary instructions.")


if page=="Symptom explorer":
    st.markdown('<div class="eyebrow">HEALTHCARE · MACHINE LEARNING LAB</div>',unsafe_allow_html=True)
    st.title("Explore a symptom pattern.")
    st.markdown('<div class="lead">Select from the symptoms actually present in the supplied dataset. Explore the model’s condition label, then inspect the evidence behind it.</div>',unsafe_allow_html=True)
    holdout=bundle.evaluation["holdout"]
    st.warning(f"Educational ML prototype — not medical diagnosis. Held-out accuracy is {holdout['accuracy']:.1%} ({round(holdout['accuracy']*holdout['rows'])} of {holdout['rows']} records). This model cannot provide a reliable medical conclusion.")
    left,right=st.columns([1.1,1],gap="large")
    with left:
        with st.container(border=True):
            st.subheader("01  Choose symptoms")
            st.caption("Only four symptoms were supplied. No age, sex, blood pressure, or other medical details are inferred.")
            selected=st.multiselect("Symptoms present",options=list(SYMPTOMS),format_func=lambda s:SYMPTOM_LABELS[s],key="symptoms",placeholder="Select one or more symptoms")
            st.caption("For this demo, every unselected symptom is treated as absent. Unknown symptoms are not supported.")
            confirmed=st.checkbox("I have reviewed all four symptoms; unselected means absent.",key="reviewed")
            run=st.button("Explore model prediction",type="primary",width="stretch",key="predict")
            if run:
                if not selected:
                    st.session_state.pop("prediction",None)
                    st.error("Select at least one symptom to explore a prediction.")
                elif not confirmed:
                    st.session_state.pop("prediction",None)
                    st.error("Confirm that you reviewed all four symptoms.")
                else:
                    st.session_state["prediction"]=predict(bundle,selected)
    with right:
        with st.container(border=True):
            st.subheader("What this experiment can show")
            st.write(f"A reproducible model’s ranking across {bundle.audit['included_labels']} condition labels with enough records for a small comparison study.")
            st.write(f"The source contains {bundle.audit['raw_labels']} labels. The other {bundle.audit['excluded_labels']} are outside the model’s scope and cannot be ruled out by a prediction.")
            st.caption("No confidence percentage is displayed: model scores are uncalibrated and are not the probability that you have a condition.")
    result=st.session_state.get("prediction")
    if result and (set(result["selected_symptoms"])!=set(selected) or not confirmed):
        st.session_state.pop("prediction",None)
        result=None
    if result:
        st.subheader("02  Explore the result")
        st.markdown(f'<div class="result"><div class="result-label">PREDICTED CONDITION · MODEL LABEL ONLY</div><div class="result-title">{html.escape(result["predicted_condition"])}</div><div class="result-note">Insufficient evidence for a reliable medical conclusion.</div></div>',unsafe_allow_html=True)
        a,b=st.columns([1.1,1],gap="large")
        with a:
            st.markdown("**Other labels in the model ranking**")
            st.write(" · ".join(x for x in result["ranked_conditions"] if x!=result["predicted_condition"]))
            st.caption("This is not a clinical differential diagnosis. Conditions outside the model are not considered.")
        with b:
            st.markdown("**How ambiguous is this pattern?**")
            st.write(f"{result['matching_training_rows']} training records share these four symptom values, spanning {result['matching_training_labels']} condition labels.")
            if not result["training_pattern_seen"]: st.warning("This exact symptom pattern was not seen in training.")
            st.caption("Matching records are associations in this dataset, not evidence of causation.")
        st.divider()
        show_reference(result["predicted_condition"],"prediction_"+result["predicted_condition"])

elif page=="Condition reference":
    st.markdown('<div class="eyebrow">SUPPLIED DATA · REFERENCE EXPLORER</div>',unsafe_allow_html=True)
    st.title("Inspect the source material.")
    st.markdown('<div class="lead">Browse condition coverage and the reference entries supplied with the original project. Missing descriptions stay missing.</div>',unsafe_allow_html=True)
    st.warning("Educational reference only. The supplied recommendation entries have not been clinically verified.")
    available=sorted(bundle.clean.disease.unique())
    condition=st.selectbox("Condition",available,index=available.index("Asthma"),key="reference_condition")
    info=condition_info(bundle,condition)
    show_reference(condition,"browse_"+condition)
    with st.expander("Symptom counts in the supplied condition records"):
        st.dataframe(pd.DataFrame({"Symptom":[SYMPTOM_LABELS[s] for s in SYMPTOMS],"Present in source records":[info["symptom_counts"][s] for s in SYMPTOMS],"Total profiles":[info["source_profiles"]]*4}),hide_index=True,width="stretch")
        st.caption("Dataset counts only. These are not medical definitions, prevalence estimates, or typical clinical symptoms.")

else:
    st.markdown('<div class="eyebrow">TRANSPARENT EVALUATION · REPRODUCIBLE RESULTS</div>',unsafe_allow_html=True)
    st.title("Understand the model.")
    st.markdown('<div class="lead">Duplicates and unavailable inputs were removed. Classifiers were compared on training data before the held-out set was evaluated.</div>',unsafe_allow_html=True)
    ev=bundle.evaluation; au=bundle.audit
    a,b,c,d=st.columns(4)
    a.metric("Held-out accuracy",f"{ev['holdout']['accuracy']:.1%}")
    b.metric("Held-out macro F1",f"{ev['holdout']['macro_f1']:.3f}")
    c.metric("Study records",au["study_rows"])
    d.metric("Condition labels",au["included_labels"])
    relationship="lower than" if ev['holdout']['accuracy']<ev['dummy_holdout']['accuracy'] else "equal to or higher than"
    st.error(f"Not suitable for medical use. The selected model’s holdout accuracy is {relationship} the simple majority baseline ({ev['dummy_holdout']['accuracy']:.1%}). It was selected by training cross-validation macro F1, not by its test score.")
    st.subheader("A small dataset, honestly evaluated")
    st.write(f"{au['raw_rows']} original rows → {au['clean_rows']} distinct records → {au['study_rows']} records across {au['included_labels']} eligible labels. {ev['train_rows']} records were used for model development; {ev['test_rows']} were reserved for the final test.")
    st.write("Records sharing all eight raw profile fields stay in one partition. All preprocessing and model selection stay inside training folds. Patient identities were not supplied, so patient independence cannot be verified.")
    st.caption("Identical four-symptom patterns still occur across partitions because different raw profiles can share symptoms. A separate training-only stress test holds out entire symptom patterns.")
    st.image(str(ROOT/"reports/figures/model_comparison.png"),width="stretch")
    comparison=pd.read_csv(ROOT/"reports/model_comparison.csv")
    st.dataframe(comparison[["model","cv_macro_f1_mean","cv_accuracy_mean"]].rename(columns={"model":"Classifier","cv_macro_f1_mean":"CV macro F1","cv_accuracy_mean":"CV accuracy"}),hide_index=True,width="stretch")
    with st.expander("Held-out evaluation details"):
        st.json({"selected_model":ev["selected_model"],"holdout":ev["holdout"],"baseline":ev["dummy_holdout"],"accuracy_95_interval":ev["holdout_accuracy_cluster_bootstrap_95"],"profile_group_overlap":0,"test_missing_labels":ev["test_missing_labels"]})
        st.caption(ev["bootstrap_note"])
        st.image(str(ROOT/"reports/figures/confusion_matrix.png"),width="stretch")
    with st.expander("Condition coverage and excluded labels"):
        st.dataframe(bundle.scope,hide_index=True,width="stretch")
    with st.expander("Removed features and source limitations"):
        st.json(au["excluded_features"])
        st.write("The notebook’s Training.csv and six supporting CSVs, plus the drug-similarity notebook’s medicine.csv, were unavailable. Their stored output is not a reproducible dataset or verified medical evidence.")

st.markdown('<div class="footer-note">For real health concerns, speak with a qualified healthcare professional. For an emergency, contact your local emergency services. This prototype is not a triage service.</div>',unsafe_allow_html=True)

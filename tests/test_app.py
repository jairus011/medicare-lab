"""Real Streamlit execution through its supported AppTest interface."""

import itertools
from pathlib import Path

from streamlit.testing.v1 import AppTest

from healthcare.data import SYMPTOMS
from healthcare.inference import load_bundle,predict

ROOT=Path(__file__).resolve().parents[1]


def launch():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
    assert not app.exception
    return app


def test_validation_and_stale_result_removal():
    at=launch()
    at.button(key="predict").click().run()
    assert any("Select at least one" in e.value for e in at.error)
    at.multiselect(key="symptoms").set_value(["cough"]).run()
    at.button(key="predict").click().run()
    assert any("Confirm" in e.value for e in at.error)
    at.checkbox(key="reviewed").check().run()
    at.button(key="predict").click().run()
    assert at.session_state["prediction"]["predicted_condition"]
    at.multiselect(key="symptoms").set_value(["fatigue"]).run()
    assert "prediction" not in at.session_state
    assert not at.exception


def test_all_15_patterns_in_streamlit():
    at=launch(); bundle=load_bundle(ROOT)
    at.checkbox(key="reviewed").check().run()
    for bits in list(itertools.product([0,1],repeat=4))[1:]:
        selected=[s for s,b in zip(SYMPTOMS,bits) if b]
        at.multiselect(key="symptoms").set_value(selected).run()
        at.button(key="predict").click().run()
        assert not at.exception
        assert at.session_state["prediction"]["predicted_condition"]==predict(bundle,selected)["predicted_condition"]
        assert any("not medical diagnosis" in item.value for item in at.warning)


def test_reference_present_missing_and_withheld():
    at=launch()
    at.radio(key="navigation").set_value("Condition reference").run()
    at.selectbox(key="reference_condition").set_value("Asthma").run()
    at.checkbox(key="references_browse_Asthma").check().run()
    text=" ".join(item.value for item in at.text)
    assert "Albuterol (Rescue Inhaler)" in text
    assert "2 puffs" not in text and "110mcg" not in text
    at.selectbox(key="reference_condition").set_value("Osteoporosis").run()
    assert any("No medicine" in item.value for item in at.info)
    at.selectbox(key="reference_condition").set_value("Stroke").run()
    assert any("conflicting emergency" in item.value for item in at.warning)
    assert not any("Aspirin" in item.value for item in at.text)
    at.selectbox(key="reference_condition").set_value("Acne").run()
    assert any("Outside the model scope" in item.value for item in at.caption)
    assert not at.exception


def test_model_data_page():
    at=launch()
    at.radio(key="navigation").set_value("Model & data").run()
    assert not at.exception
    assert at.metric[0].value=="5.3%"
    assert len(at.dataframe)>=2
    assert any("majority baseline" in item.value for item in at.error)

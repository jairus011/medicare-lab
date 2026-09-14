"""Build a traceable display subset of the supplied medicine database.

No doses, schedules, treatment selection, or emergency medication instructions
are published by the application. No medical text is generated here.
"""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/reference/medicine_database.source.json"

# Manually reviewed exact substrings of the original medicine entries.
NAME_SELECTIONS = {
    "Influenza": [(0,"Oseltamivir (Tamiflu)"),(1,"Acetaminophen"),(2,"Ibuprofen")],
    "Asthma": [(0,"Albuterol (Rescue Inhaler)"),(1,"Fluticasone (Controller)"),(2,"Montelukast")],
    "Diabetes": [(0,"Metformin"),(1,"Insulin (if prescribed)")],
    "Hypertension": [(0,"Lisinopril"),(1,"Amlodipine"),(2,"Losartan")],
    "Pneumonia": [(0,"Amoxicillin"),(1,"Azithromycin"),(2,"Acetaminophen")],
    "Common Cold": [(0,"Acetaminophen"),(1,"Pseudoephedrine"),(2,"Vitamin C")],
    "Bronchitis": [(0,"Albuterol inhaler"),(1,"Prednisone"),(2,"Dextromethorphan cough syrup")],
    "Depression": [(0,"Sertraline (Zoloft)"),(1,"Fluoxetine (Prozac)")],
    "Anxiety Disorders": [(0,"Sertraline"),(1,"Buspirone"),(2,"Alprazolam")],
}


def build():
    source=json.loads(SOURCE.read_text(encoding="utf-8"))
    records={}
    for condition, fields in source.items():
        record={"condition":condition,"source":"medicare.zip / medicine_database.pkl",
            "verified_medically":False,"description":None,
            "description_status":"No disease description was supplied in the available reference dataset.",
            "medicine_names":[],"foods_to_eat":[],"foods_to_avoid":[],"source_reported_course":None,
            "withheld":condition=="Stroke"}
        if record["withheld"]:
            record["withheld_reason"]="The source contains conflicting emergency medication instructions. Its recommendation fields are withheld from this educational interface."
        else:
            for index,name in NAME_SELECTIONS.get(condition,[]):
                original=fields["medicines"][index]
                if name not in original: raise ValueError(f"Ungrounded reference: {condition}: {name}")
                record["medicine_names"].append({"text":name,"source_field":"medicines","source_index":index,
                    "source_text_sha256":hashlib.sha256(original.encode()).hexdigest()})
            for key in ("foods_to_eat","foods_to_avoid"):
                record[key]=[{"text":text,"source_field":key,"source_index":i} for i,text in enumerate(fields.get(key,[]))]
            record["source_reported_course"]={"text":fields["recovery_time"],"source_field":"recovery_time"}
        records[condition]=record
    target=ROOT/"data/reference/display_reference.json"
    payload={"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "policy":"Unverified historical source entries for education only. Names are substrings; food and course entries are verbatim. Doses, schedules, advice, and triage instructions are omitted. No invented fallback content.",
        "records":records}
    target.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Prepared {len(records)} source-linked condition records")


if __name__ == "__main__":
    build()

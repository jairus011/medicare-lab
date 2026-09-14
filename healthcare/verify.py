"""Fail deployment early if the bundled model, data, or runtime is incompatible."""

from pathlib import Path

from .inference import load_bundle, predict


def main():
    bundle = load_bundle(Path(__file__).resolve().parents[1])
    result = predict(bundle, ["cough"])
    if result["predicted_condition"] not in bundle.metadata["labels"]:
        raise ValueError("The bundled model returned an unexpected label")
    if not result["educational_only"]:
        raise ValueError("The educational inference contract was not preserved")
    print("Artifact integrity, runtime compatibility, and inference checks passed.")


if __name__ == "__main__":
    main()

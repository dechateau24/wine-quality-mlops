import tempfile
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "models" / "reference.csv"


def build_current_data(reference):
    """Simule des donnees de production : memes vins, mais mesures decalees."""
    current = reference.sample(n=500, random_state=0).copy()
    current["alcohol"] = current["alcohol"] + 1.5
    current["volatile.acidity"] = current["volatile.acidity"] * 1.4
    return current


def generate_report_html():
    reference = pd.read_csv(REFERENCE_PATH)
    current = build_current_data(reference)

    definition = DataDefinition(numerical_columns=list(reference.columns))
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)
    current_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(current_ds, reference_ds)  # ordre : courant, puis reference

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drift_report.html"
        result.save_html(str(path))
        return path.read_text(encoding="utf-8")
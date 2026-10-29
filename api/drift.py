import tempfile
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "models" / "reference.csv"

# Colonnes dont on simule la derive (7 sur 11, soit plus de 50 %)
DRIFTED_COLUMNS = [
    "alcohol",
    "volatile.acidity",
    "density",
    "chlorides",
    "residual.sugar",
    "total.sulfur.dioxide",
    "pH",
]


def build_current_data(reference):
    """Simule des donnees de production : memes vins, mais 7 mesures decalees."""
    current = reference.sample(n=500, random_state=0).copy()
    for col in DRIFTED_COLUMNS:
        current[col] = current[col] + reference[col].std()
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
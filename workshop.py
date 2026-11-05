"""Local-testing helpers for the workshop.

Nothing in here is part of an analysis you would upload to FLAME. It only
prepares the synthetic data so that ``StarModelTester`` can simulate the three
hospitals on your laptop.
"""

import contextlib
import io
import pickle
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
HOSPITALS = ["hospital_a", "hospital_b", "hospital_c"]
FILENAME = "patients.csv"


def load_hospitals():
    """Return one data split per hospital, in the shape a FLAME node delivers it.

    A node hands ``analysis_method`` a list with one dict per data store that is
    connected to the project. For an S3 (object storage) data store, that dict
    maps each file name to the file's raw bytes. Every simulated hospital here
    has one data store holding one file::

        [{"patients.csv": b"pseudonym,sex,age,..."}]
    """
    return [[{FILENAME: (DATA_DIR / f"{hospital}.csv").read_bytes()}] for hospital in HOSPITALS]


def pooled_dataframe():
    """All hospitals in one DataFrame: the central data pool that FLAME avoids.

    This only works here because the data is synthetic and on your laptop. Use
    it to check that a federated result equals what a central analysis of the
    pooled data would have given.
    """
    import pandas as pd

    return pd.concat([pd.read_csv(DATA_DIR / f"{hospital}.csv") for hospital in HOSPITALS], ignore_index=True)


def run_quietly(**tester_kwargs):
    """Run ``StarModelTester`` without its log output and return the final result.

    Handy for repeating an analysis many times, e.g. to look at the spread of
    differentially private results. If the run fails, its logs are printed.
    """
    from flame.star import StarModelTester

    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "result.pkl"
        logs = io.StringIO()
        with contextlib.redirect_stdout(logs):
            StarModelTester(**tester_kwargs, output_type="pickle", filename=str(result_file))
        if not result_file.exists():
            print(logs.getvalue())
            raise RuntimeError("The simulated analysis failed, see the logs above.")
        return pickle.loads(result_file.read_bytes())

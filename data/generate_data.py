"""Generate the synthetic diabetes cohorts used in the workshop.

All values are random draws; no real patient is behind any row. The three
hospitals deliberately differ (size, age, case mix), because that is what makes
federated statistics interesting - and hospital C is deliberately small, so
that some subgroups fall below typical disclosure thresholds.

Run from the project root:  python data/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SEED = 2026

# name, number of patients, mean age, case mix (T2D, T1D, prediabetes, none), zip code prefix,
# HbA1c shift for T2D patients (worse glycaemic control at hospital C)
SITES = [
    ("hospital_a", 480, 58, (0.30, 0.08, 0.20, 0.42), "72", 0.0),
    ("hospital_b", 650, 52, (0.22, 0.15, 0.18, 0.45), "80", -0.2),
    ("hospital_c", 37, 67, (0.45, 0.05, 0.20, 0.30), "01", 0.5),
]

TYPES = ["T2D", "T1D", "prediabetes", "none"]
ICD10 = {"T2D": "E11.9", "T1D": "E10.9", "prediabetes": "R73.0", "none": ""}


def generate_site(rng, name, n, mean_age, mix, zip_prefix, t2d_shift):
    """Draw one hospital's cohort."""
    diabetes_type = rng.choice(TYPES, size=n, p=mix)
    is_t1d = diabetes_type == "T1D"
    is_t2d = diabetes_type == "T2D"
    is_pre = diabetes_type == "prediabetes"

    age = np.where(is_t1d, rng.normal(36, 13, n), rng.normal(mean_age, 12, n))
    age = np.clip(np.round(age), 18, 95).astype(int)
    sex = np.where(rng.random(n) < np.where(is_t2d, 0.56, 0.48), "male", "female")

    bmi = np.select(
        [is_t2d, is_t1d, is_pre],
        [rng.normal(31.0, 5.0, n), rng.normal(25.0, 3.5, n), rng.normal(29.0, 4.5, n)],
        rng.normal(26.0, 4.0, n),
    )
    bmi = np.round(np.clip(bmi, 16, 55), 1)

    hba1c = np.select(
        [is_t2d, is_t1d, is_pre],
        [rng.normal(7.4 + t2d_shift, 1.1, n), rng.normal(7.9, 1.0, n), rng.uniform(5.7, 6.4, n)],
        rng.normal(5.3, 0.25, n),
    )
    hba1c = np.round(np.clip(hba1c, 4.5, 14.0), 1)
    # Routine data: HbA1c is mostly measured when there is a reason to.
    not_measured = (diabetes_type == "none") & (rng.random(n) < 0.35)

    # Estimated average glucose (ADAG) as a rough stand-in for fasting glucose.
    fasting_glucose = np.round(28.7 * hba1c - 46.7 + rng.normal(0, 12, n)).astype(int)
    systolic_bp = np.round(118 + 0.45 * (age - 50) + 0.8 * (bmi - 25) + rng.normal(0, 12, n)).astype(int)
    smoker = (rng.random(n) < np.where(is_t2d, 0.32, 0.20)).astype(int)

    df = pd.DataFrame(
        {
            "pseudonym": [f"{name[-1].upper()}-{rng.integers(16**6):06X}" for _ in range(n)],
            "sex": sex,
            "age": age,
            "zip_code": [f"{zip_prefix}{rng.integers(1000):03d}" for _ in range(n)],
            "diabetes_type": diabetes_type,
            "icd10_code": [ICD10[t] for t in diabetes_type],
            "bmi": bmi,
            "hba1c": hba1c,
            "fasting_glucose": fasting_glucose,
            "systolic_bp": systolic_bp,
            "smoker": smoker,
        }
    )
    df["hba1c"] = df["hba1c"].mask(not_measured)
    df["fasting_glucose"] = df["fasting_glucose"].mask(not_measured).astype("Int64")
    return df


def main():
    rng = np.random.default_rng(SEED)
    for name, *params in SITES:
        df = generate_site(rng, name, *params)
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} patients -> data/{name}.csv")


if __name__ == "__main__":
    main()

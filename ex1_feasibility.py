"""Exercise 1 - Your own feasibility query (about 10 min)

A feasibility query answers "Do the hospitals have enough patients for my
study?" before anybody applies for access to the data itself.

Tasks
  a) Change ``in_cohort`` to the inclusion criteria of your own study idea, for
     example "T2D, aged 40-75, HbA1c >= 8 % and BMI >= 30". Run the script.
  b) Hospital C is small. A count of 1 or 2 can point to a real person ("the
     only 30-year-old with T1D in that town"). Add small-cell suppression: a
     hospital whose count is below MIN_CELL_SIZE must not report the exact
     number. Look for the two TODO b) markers.
  c) Discuss with your neighbour: is the total still safe? What could someone
     learn by sending two almost identical queries (a "differencing attack")?

Columns in patients.csv: pseudonym, sex, age, zip_code, diabetes_type
(T2D / T1D / prediabetes / none), icd10_code, bmi, hba1c (%), fasting_glucose
(mg/dl), systolic_bp (mmHg), smoker (0/1). hba1c is empty when it was never
measured.
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

MIN_CELL_SIZE = 5


def in_cohort(patients):
    """Return a True/False mask: which patients meet the inclusion criteria."""
    # TODO a) replace with your own criteria
    return (patients["diabetes_type"] == "T2D") & (patients["age"] >= 18)


class FeasibilityAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        count = int(in_cohort(patients).sum())
        # TODO b) return None instead of a count below MIN_CELL_SIZE
        return count


class FeasibilityAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        # TODO b) suppressed hospitals send None: leave them out of the total,
        #         but report how many hospitals were suppressed
        return {"per_hospital": analysis_results, "total": sum(analysis_results)}

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    from workshop import load_hospitals

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=FeasibilityAnalyzer,
        aggregator=FeasibilityAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

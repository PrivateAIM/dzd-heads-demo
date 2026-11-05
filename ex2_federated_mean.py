"""Exercise 2 - Federated mean and standard deviation (about 10 min)

Question: what is the mean HbA1c, and its standard deviation, of all T2D
patients across the three hospitals?

Tasks
  a) Run the script. Every hospital sends its own mean HbA1c, and the
     aggregator averages those means. Compare the result with the pooled
     reference printed at the end. Why is it wrong? (Hint: hospital C.)
  b) Fix it. Let every hospital send its "sufficient statistics" instead: the
     number of values n, their sum and their sum of squares. Compute the pooled
     mean and SD in the aggregator. Your numbers should match the reference.
     Formulas:  mean = sum / n    variance = (sum_sq - sum**2 / n) / (n - 1)
  c) Discuss: do n, sum and sum of squares reveal more than a mean does? What
     if a hospital has n = 1?

Note: hba1c is empty for patients where it was never measured, which is
typical for routine data. Use the number of measured values for n, not the
number of patients.
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester


class MeanAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        values = patients.loc[patients["diabetes_type"] == "T2D", "hba1c"].dropna()
        # TODO b) return n, sum and sum of squares instead of the local mean
        return float(values.mean())


class MeanAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        # TODO b) pool the sufficient statistics into the overall mean and SD
        return {"mean_hba1c": round(sum(analysis_results) / len(analysis_results), 4)}

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    from workshop import load_hospitals, pooled_dataframe

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=MeanAnalyzer,
        aggregator=MeanAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

    # Reference: what a central analysis of the pooled data would give.
    pooled = pooled_dataframe()
    reference = pooled.loc[pooled["diabetes_type"] == "T2D", "hba1c"].dropna()
    print(f"\nPooled reference: mean_hba1c={reference.mean():.4f}  sd_hba1c={reference.std():.4f}  n={len(reference)}")

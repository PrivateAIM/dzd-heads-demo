"""Solution to exercise 2 - pooled mean and SD from sufficient statistics.

Averaging the hospital means weights tiny hospital C (17 T2D patients, worse
HbA1c) as much as the two large ones. Sending n, sum and sum of squares
gives exactly the pooled result.

Discussion for c): with n = 1, the sum *is* that patient's value. Sufficient
statistics need the same minimum cell size as counts do.
"""

import io
import math

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester


class MeanAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        values = patients.loc[patients["diabetes_type"] == "T2D", "hba1c"].dropna()
        return {"n": len(values), "sum": float(values.sum()), "sum_sq": float((values**2).sum())}


class MeanAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        n = sum(result["n"] for result in analysis_results)
        total = sum(result["sum"] for result in analysis_results)
        total_sq = sum(result["sum_sq"] for result in analysis_results)
        mean = total / n
        variance = (total_sq - total**2 / n) / (n - 1)
        return {"mean_hba1c": round(mean, 4), "sd_hba1c": round(math.sqrt(variance), 4), "n": n}

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from workshop import load_hospitals, pooled_dataframe

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=MeanAnalyzer,
        aggregator=MeanAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

    pooled = pooled_dataframe()
    reference = pooled.loc[pooled["diabetes_type"] == "T2D", "hba1c"].dropna()
    print(f"\nPooled reference: mean_hba1c={reference.mean():.4f}  sd_hba1c={reference.std():.4f}  n={len(reference)}")

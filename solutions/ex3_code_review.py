"""Solution to exercise 3 - review of the "glycaemic control" analysis.

Decision: reject, or request changes. Problems:

1. Row-level data leaves the hospital. The "outliers" table contains single
   patients with pseudonym, sex, age, full zip code and HbA1c. A pseudonym is
   still personal data (GDPR Art. 4(5)), and age + sex + zip code are
   quasi-identifiers that can re-identify people even without the pseudonym.
   A data quality check like this belongs at the hospital, not in a result.
2. Patient rows are written to the logs (patients.head(3)). Node logs are
   visible to the node admins and are forwarded to the Hub by default.
3. Small cells. Age groups with 1-4 patients are reported, e.g. "81+" with
   count 1 at one hospital, so its mean is one patient's value.
4. min and max are always one real patient's value, whatever the cell size.
5. The aggregator does not aggregate. It passes every hospital's result
   through, so the researcher receives hospital-level (and row-level) data.
6. Hidden network call. report_usage() sends data to an external server. It
   is switched off today, but code that exists can be switched on in the
   next version. FLAME analysis pods have no internet access, so it would
   fail anyway. Defence in depth: review *and* isolation.

The fixed version below sends per-hospital counts and sums by age group,
suppresses small cells, logs nothing about patients, and pools the groups in
the aggregator.
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

MIN_CELL_SIZE = 5
AGE_BINS = [17, 40, 60, 80, 120]
AGE_LABELS = ["18-40", "41-60", "61-80", "81+"]


class GlycaemicControlAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        t2d = patients[(patients["diabetes_type"] == "T2D") & patients["hba1c"].notna()].copy()
        t2d["age_group"] = pd.cut(t2d["age"], bins=AGE_BINS, labels=AGE_LABELS)

        groups = {}
        for age_group, values in t2d.groupby("age_group", observed=False)["hba1c"]:
            if len(values) >= MIN_CELL_SIZE:
                groups[age_group] = {"n": len(values), "sum": float(values.sum())}
        self.flame.flame_log(f"Reporting {len(groups)} of {len(AGE_LABELS)} age groups")
        return groups


class PoolingAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        pooled = {}
        for age_group in AGE_LABELS:
            reported = [result[age_group] for result in analysis_results if age_group in result]
            n = sum(group["n"] for group in reported)
            if n >= MIN_CELL_SIZE:
                pooled[age_group] = {"n": n, "mean_hba1c": round(sum(group["sum"] for group in reported) / n, 2)}
            else:
                pooled[age_group] = f"suppressed (<{MIN_CELL_SIZE})"
        return pooled

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from workshop import load_hospitals

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=GlycaemicControlAnalyzer,
        aggregator=PoolingAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

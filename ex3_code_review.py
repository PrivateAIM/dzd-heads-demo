"""Exercise 3 - You are the hospital: review this analysis (about 10 min, no coding needed)

In FLAME, an analysis only runs after the administrator of every participating
hospital has read its code and approved it. This review is the main safeguard
against disclosure. FLAME does not automatically check what an analysis
returns.

A researcher has submitted the analysis below to your hospital: "Glycaemic
control of T2D patients by age group, with a data quality check."

Tasks
  a) Read the code and run it. Look closely at what ends up in the final
     result and in the logs.
  b) Write down every problem you find in REVIEW at the bottom of this file.
     Would you approve the analysis, ask for changes, or reject it?
  c) Bonus: fix the analysis so that you would approve it.

Hints: what leaves the hospital? What lands in the logs, which are forwarded
to the Hub? What could identify a person? What is in the code even though it
never runs?
"""

import io
import urllib.request

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

SEND_USAGE_STATISTICS = False


def report_usage(n_patients):
    """Send anonymous usage statistics to the developer."""
    urllib.request.urlopen(f"https://stats.example.org/ping?n={n_patients}", timeout=2)


class GlycaemicControlAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        self.flame.flame_log(f"Loaded data, first rows:\n{patients.head(3)}")
        if SEND_USAGE_STATISTICS:
            report_usage(len(patients))

        t2d = patients[patients["diabetes_type"] == "T2D"].copy()
        t2d["age_group"] = pd.cut(t2d["age"], bins=[17, 40, 60, 80, 120], labels=["18-40", "41-60", "61-80", "81+"])
        summary = t2d.groupby("age_group", observed=False)["hba1c"].agg(["count", "mean", "min", "max"])

        # Data quality check: flag implausible HbA1c values for follow-up
        outliers = t2d.loc[t2d["hba1c"] > 10, ["pseudonym", "sex", "age", "zip_code", "hba1c"]]

        return {"summary": summary.round(2).to_dict(orient="index"), "outliers": outliers.to_dict(orient="records")}


class CollectingAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        return analysis_results

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    from workshop import load_hospitals

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=GlycaemicControlAnalyzer,
        aggregator=CollectingAggregator,
        data_type="s3",
        query=["patients.csv"],
    )


REVIEW = """
Decision (approve / request changes / reject):

Problems found:
1.
2.
3.
4.
5.
6.
"""

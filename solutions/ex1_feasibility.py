"""Solution to exercise 1 - feasibility query with small-cell suppression.

Discussion for c): suppression per hospital is not enough on its own. If the
total over all hospitals is published, and the same query is also answered
without hospital C (or with a slightly wider age range), the difference
reveals the suppressed number. Real feasibility portals therefore also round
or perturb the counts they return, and they limit or log how many queries
somebody may send.
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

MIN_CELL_SIZE = 5


def in_cohort(patients):
    """Return a True/False mask: which patients meet the inclusion criteria."""
    return (
        (patients["diabetes_type"] == "T2D")
        & patients["age"].between(40, 75)
        & (patients["hba1c"] >= 8.0)
        & (patients["bmi"] >= 30)
    )


class FeasibilityAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        count = int(in_cohort(patients).sum())
        if count < MIN_CELL_SIZE:
            self.flame.flame_log(f"Count below {MIN_CELL_SIZE}, suppressed")
            return None
        return count


class FeasibilityAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        reported = [count for count in analysis_results if count is not None]
        return {
            "per_hospital": [f"<{MIN_CELL_SIZE}" if count is None else count for count in analysis_results],
            "total_of_reporting_hospitals": sum(reported),
            "hospitals_suppressed": len(analysis_results) - len(reported),
        }

    def has_converged(self, result, last_result):
        return True


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from workshop import load_hospitals

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=FeasibilityAnalyzer,
        aggregator=FeasibilityAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

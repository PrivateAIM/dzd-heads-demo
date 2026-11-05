"""Demo: a federated feasibility query.

Question: "How many patients with type 2 diabetes do the three hospitals have?"

Every hospital counts its own patients. Only that count leaves the hospital,
and the aggregator adds the counts up.

Everything above the ``if __name__ == "__main__":`` line is what you would
upload to the FLAME Hub. The block below it only simulates the hospitals on
your laptop.
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester


class CountAnalyzer(StarAnalyzer):
    """Runs inside every hospital, next to the data."""

    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        count = int((patients["diabetes_type"] == "T2D").sum())
        self.flame.flame_log(f"{len(patients)} patients in this hospital, {count} with T2D")
        return count


class SumAggregator(StarAggregator):
    """Runs on the aggregator node. It never sees patient data, only the counts."""

    def aggregation_method(self, analysis_results):
        return {"per_hospital": analysis_results, "total": sum(analysis_results)}

    def has_converged(self, result, last_result):
        return True  # a single round is enough for a count


if __name__ == "__main__":
    from workshop import load_hospitals

    StarModelTester(
        data_splits=load_hospitals(),  # one data split per simulated hospital
        analyzer=CountAnalyzer,
        aggregator=SumAggregator,
        data_type="s3",
        query=["patients.csv"],
    )

"""Bonus exercise 4 - Differential privacy (about 10 min)

Exercise 1 showed that exact counts can leak, even after small-cell
suppression. Differential privacy (DP) releases a result with calibrated random
noise instead. FLAME's ``StarLocalDPModel`` adds Laplace noise with scale
``sensitivity / epsilon`` to the final numeric result before it is released.
In the local test you switch it on by passing ``epsilon`` and ``sensitivity``
to ``StarModelTester``.

Tasks
  a) Run the script. It releases the number of T2D patients five times for
     each epsilon. How far are the noisy counts from the exact count?
  b) What is the sensitivity of a count, i.e. by how much can adding or
     removing one patient change it?
  c) Change ``in_cohort`` to a rare subgroup, e.g. T1D patients aged 60 or
     older. Is the result still useful at epsilon = 0.1?
  d) Discuss: the script asks the same question five times. Why is that a
     problem for privacy? Who should choose epsilon: the researcher or the
     hospital that reviews the analysis?
"""

import io

import pandas as pd
from flame.star import StarAggregator, StarAnalyzer

EPSILONS = [0.1, 1.0, 10.0]
SENSITIVITY = 1.0
REPETITIONS = 5


def in_cohort(patients):
    """Return a True/False mask: which patients to count."""
    return patients["diabetes_type"] == "T2D"


class CountAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        return int(in_cohort(patients).sum())


class SumAggregator(StarAggregator):
    def aggregation_method(self, analysis_results):
        return sum(analysis_results)

    def has_converged(self, result, last_result):
        return True  # noise is only added once the aggregator reports convergence


if __name__ == "__main__":
    from workshop import load_hospitals, pooled_dataframe, run_quietly

    exact = int(in_cohort(pooled_dataframe()).sum())
    print(f"Exact count (pooled reference): {exact}\n")
    for epsilon in EPSILONS:
        noisy = [
            run_quietly(
                data_splits=load_hospitals(),
                analyzer=CountAnalyzer,
                aggregator=SumAggregator,
                data_type="s3",
                query=["patients.csv"],
                epsilon=epsilon,
                sensitivity=SENSITIVITY,
            )
            for _ in range(REPETITIONS)
        ]
        print(f"epsilon={epsilon:<5}  released: " + "  ".join(f"{value:7.1f}" for value in noisy))

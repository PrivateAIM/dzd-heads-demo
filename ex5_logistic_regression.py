"""Bonus exercise 5 - An iterative analysis: federated logistic regression (about 10 min)

Counts and means need one round. Model fitting needs several. Here the
aggregator sends the current coefficients to the hospitals. Every hospital
sends back the gradient and Hessian of its local log-likelihood, and the
aggregator takes one Newton step. This repeats until the coefficients stop
changing. Federated GLMs in DataSHIELD work the same way.

Model: P(T2D) ~ age (per 10 years) + BMI (per 5 kg/m2) + male + smoker

Tasks
  a) Run the script and compare the federated odds ratios with the pooled
     reference at the end. How many rounds were needed?
  b) Add systolic blood pressure (per 10 mmHg) as a feature in
     ``design_matrix`` and run it again.
  c) Discuss: what could a gradient or Hessian from a hospital with only a
     handful of patients reveal? What would help (minimum n, secure
     aggregation, DP)?
"""

import io

import numpy as np
import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester


def design_matrix(patients):
    """Return the feature matrix X (with intercept) and the outcome y (T2D yes/no)."""
    features = pd.DataFrame(
        {
            "intercept": 1.0,
            "age_per_10y": (patients["age"] - 50) / 10,
            "bmi_per_5": (patients["bmi"] - 27) / 5,
            "male": (patients["sex"] == "male").astype(float),
            "smoker": patients["smoker"].astype(float),
        }
    )
    outcome = (patients["diabetes_type"] == "T2D").astype(float)
    return features, outcome


class NewtonAnalyzer(StarAnalyzer):
    def analysis_method(self, data, aggregator_results):
        patients = pd.read_csv(io.BytesIO(data[0]["patients.csv"]))
        features, outcome = design_matrix(patients)
        x, y = features.to_numpy(), outcome.to_numpy()
        if aggregator_results is None:  # first round: start from all-zero coefficients
            beta = np.zeros(x.shape[1])
        else:
            beta = np.array(aggregator_results["coefficients"])
        p = 1 / (1 + np.exp(-x @ beta))
        gradient = x.T @ (y - p)
        hessian = (x * (p * (1 - p))[:, None]).T @ x
        return {"names": list(features.columns), "gradient": gradient.tolist(), "hessian": hessian.tolist()}


class NewtonAggregator(StarAggregator):
    def __init__(self, flame):
        super().__init__(flame)
        self.beta = None

    def aggregation_method(self, analysis_results):
        names = analysis_results[0]["names"]
        gradient = sum(np.array(result["gradient"]) for result in analysis_results)
        hessian = sum(np.array(result["hessian"]) for result in analysis_results)
        if self.beta is None:
            self.beta = np.zeros(len(names))
        self.beta = self.beta + np.linalg.solve(hessian, gradient)
        self.flame.flame_log(f"Round {self.num_iterations}: coefficients {np.round(self.beta, 4).tolist()}")
        return {
            "coefficients": self.beta.tolist(),
            "odds_ratios": {name: round(float(np.exp(b)), 3) for name, b in zip(names[1:], self.beta[1:])},
        }

    def has_converged(self, result, last_result):
        if last_result is None:
            return False
        change = np.abs(np.array(result["coefficients"]) - np.array(last_result["coefficients"]))
        return bool(change.max() < 1e-8) or self.num_iterations >= 25


if __name__ == "__main__":
    from workshop import load_hospitals, pooled_dataframe

    StarModelTester(
        data_splits=load_hospitals(),
        analyzer=NewtonAnalyzer,
        aggregator=NewtonAggregator,
        data_type="s3",
        query=["patients.csv"],
        simple_analysis=False,  # several rounds until has_converged returns True
    )

    # Reference: the same Newton iterations on the pooled data.
    features, outcome = design_matrix(pooled_dataframe())
    x, y, beta = features.to_numpy(), outcome.to_numpy(), np.zeros(features.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-x @ beta))
        beta = beta + np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x, x.T @ (y - p))
    odds_ratios = {name: round(float(np.exp(b)), 3) for name, b in zip(features.columns[1:], beta[1:])}
    print(f"\nPooled reference odds ratios: {odds_ratios}")

"""
generate_sample_data.py
Generates a realistic synthetic hiring dataset to demo FairLens.
Contains intentional bias to showcase detection capabilities.

Run: python generate_sample_data.py
Output: sample_hiring_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# Demographics
gender     = np.random.choice(["Male", "Female", "Non-binary"], N, p=[0.55, 0.40, 0.05])
race       = np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"], N, p=[0.55, 0.18, 0.14, 0.10, 0.03])
age        = np.random.randint(22, 65, N)

# Proxy variables (correlated with protected attributes)
# ZIP code as proxy for race
zipcode_base = {"White": 10001, "Black": 10025, "Hispanic": 10031, "Asian": 10013, "Other": 10044}
zipcode = np.array([zipcode_base[r] + np.random.randint(-3, 4) for r in race])

# School tier — correlated with race/income (proxy discrimination)
school_probs = {
    "White":    [0.35, 0.40, 0.25],
    "Black":    [0.10, 0.35, 0.55],
    "Hispanic": [0.08, 0.32, 0.60],
    "Asian":    [0.45, 0.40, 0.15],
    "Other":    [0.20, 0.40, 0.40],
}
school_tier = np.array([np.random.choice(["Top", "Mid", "Low"], p=school_probs[r]) for r in race])

# Years of experience
experience = np.clip(np.random.normal(8, 4, N), 0, 30).astype(int)

# Test score (slight gender gap introduced for bias demo)
base_score = np.random.normal(70, 12, N)
gender_penalty = np.where(gender == "Female", -3, 0)  # artificial bias
test_score = np.clip(base_score + gender_penalty, 0, 100).astype(int)

# Salary history (income proxy)
salary_history = np.random.normal(65000, 20000, N).astype(int)
# Encode racial income disparity (real-world pattern)
salary_history += np.where(race == "White", 8000, 0)
salary_history += np.where(race == "Asian", 5000, 0)
salary_history = np.clip(salary_history, 25000, 200000)

# Hiring outcome — biased model
def compute_hire_prob(g, r, a, exp, score, school, sal):
    p = 0.30
    p += 0.003 * exp
    p += 0.004 * (score - 70)
    p += {"Top": 0.20, "Mid": 0.05, "Low": -0.05}[school]
    # Intentional gender bias
    if g == "Female": p -= 0.08
    # Intentional racial bias (subtle)
    if r == "Black":    p -= 0.06
    if r == "Hispanic": p -= 0.04
    return np.clip(p, 0.02, 0.98)

hire_probs = np.array([
    compute_hire_prob(gender[i], race[i], age[i], experience[i], test_score[i], school_tier[i], salary_history[i])
    for i in range(N)
])
hired = (np.random.rand(N) < hire_probs).astype(int)

df = pd.DataFrame({
    "gender":          gender,
    "race":            race,
    "age":             age,
    "zipcode":         zipcode,
    "school_tier":     school_tier,
    "years_experience": experience,
    "test_score":      test_score,
    "salary_history":  salary_history,
    "hired":           hired,
})

df.to_csv("sample_hiring_dataset.csv", index=False)

print(f"✅ Generated sample_hiring_dataset.csv ({N} rows)")
print("\nClass distribution:")
print(df["hired"].value_counts())
print("\nHire rate by gender:")
print(df.groupby("gender")["hired"].mean().round(3))
print("\nHire rate by race:")
print(df.groupby("race")["hired"].mean().round(3))
print("\n💡 Upload sample_hiring_dataset.csv to FairLens and set target column to 'hired'")

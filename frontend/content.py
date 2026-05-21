"""Copy, glossary, and preset patients for the Streamlit UI."""

EXAMPLE_PATIENTS = {
    "Custom (use sliders)": None,
    "Low risk — younger, healthy": {"age": 45, "blood_pressure": 115, "comorbidities": 0},
    "Typical — average profile": {"age": 65, "blood_pressure": 120, "comorbidities": 2},
    "High risk — older, complex": {"age": 82, "blood_pressure": 145, "comorbidities": 5},
}

GLOSSARY = {
    "CATE (Individual effect)": (
        "Conditional Average Treatment Effect — how much the treatment is expected to "
        "**help or harm this specific patient**, in outcome score units."
    ),
    "ATE (Population effect)": (
        "Average Treatment Effect — the **average benefit** you would expect across patients "
        "similar to those in the training data."
    ),
    "Treat / Withhold": (
        "**Treat (T=1)** means give the intervention. **Withhold (T=0)** means do not. "
        "The policy recommends whichever side has the better expected outcome."
    ),
    "Causal DAG": (
        "A directed graph showing **which variables may influence which**. "
        "Arrows are learned from data (NOTEARS) or shown as a **reference** from clinical knowledge."
    ),
    "Counterfactual": (
        "A **what-if** scenario: “What outcome do we expect if we treat vs if we withhold?”"
    ),
    "SHAP (Why?)": (
        "Shows **which patient features push the estimated treatment effect up or down** "
        "for this individual."
    ),
}

HOW_TO_USE = """
1. **Choose a patient** — pick an example profile or adjust age, blood pressure, and comorbidities.  
2. Click **Run Analysis**.  
3. Read the **summary**, then explore **Recommendation → Why → What-if** tabs.  
4. Use the **Glossary** anytime a term is unclear.
"""

DISCLAIMER = (
    "This tool estimates treatment effects from **observational patterns** in a synthetic cohort. "
    "It supports discussion and exploration — **it does not replace clinical judgment** or "
    "regulatory approval processes."
)

ABOUT_ANALYSIS = """
| Topic | Detail |
|-------|--------|
| **Data** | Synthetic MIMIC-style cohort (~2,000 patients), not this individual’s real EHR |
| **Effect model** | Doubly robust learner (EconML) with overlap-aware propensity |
| **DAG** | NOTEARS on normalised observational data; sparse edges are possible |
| **Bandit** | LinUCB contextual bandit (exploratory second opinion on treatment choice) |
| **Limitations** | Unmeasured confounding, extrapolation outside training data, illustrative confidence |
"""

MAGNITUDE_LABELS = {
    "small": "Small expected impact",
    "moderate": "Moderate expected impact",
    "large": "Large expected impact",
}

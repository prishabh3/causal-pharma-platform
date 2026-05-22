import type { DagResponse } from "./types";

export const EXAMPLE_PATIENTS: Record<string, { age: number; blood_pressure: number; comorbidities: number } | null> = {
  "Custom (use sliders)": null,
  "Low risk — younger, healthy": { age: 45, blood_pressure: 115, comorbidities: 0 },
  "Typical — average profile": { age: 65, blood_pressure: 120, comorbidities: 2 },
  "High risk — older, complex": { age: 82, blood_pressure: 145, comorbidities: 5 },
};

export const GLOSSARY: Record<string, string> = {
  "CATE (Individual effect)":
    "Conditional Average Treatment Effect — how much the treatment is expected to help or harm this specific patient, in outcome score units.",
  "ATE (Population effect)":
    "Average Treatment Effect — the average benefit you would expect across patients similar to those in the training data.",
  "Treat / Withhold":
    "Treat (T=1) means give the intervention. Withhold (T=0) means do not. The policy recommends whichever side has the better expected outcome.",
  "Causal DAG":
    "A directed graph showing which variables may influence which. Arrows are learned from data (NOTEARS) or shown as a reference from clinical knowledge.",
  Counterfactual:
    "A what-if scenario: 'What outcome do we expect if we treat vs if we withhold?'",
  "SHAP (Why?)":
    "Shows which patient features push the estimated treatment effect up or down for this individual.",
};

export const ABOUT_ANALYSIS = `
| Topic | Detail |
|-------|--------|
| **Data** | Synthetic MIMIC-style cohort (~2,000 patients), not this individual's real EHR |
| **Effect model** | Doubly robust learner (EconML) with overlap-aware propensity |
| **DAG** | NOTEARS on normalised observational data; sparse edges are possible |
| **Bandit** | LinUCB contextual bandit (exploratory second opinion on treatment choice) |
| **Limitations** | Unmeasured confounding, extrapolation outside training data, illustrative confidence |
`;

export const MAGNITUDE_LABELS: Record<string, string> = {
  small: "Small expected impact",
  moderate: "Moderate expected impact",
  large: "Large expected impact",
};

export const REFERENCE_DAG_FALLBACK: DagResponse = {
  nodes: ["age", "blood_pressure", "comorbidities", "treatment", "outcome"],
  edges: [
    { source: "age", target: "treatment" },
    { source: "blood_pressure", target: "treatment" },
    { source: "comorbidities", target: "treatment" },
    { source: "age", target: "outcome" },
    { source: "blood_pressure", target: "outcome" },
    { source: "comorbidities", target: "outcome" },
    { source: "treatment", target: "outcome" },
  ],
};

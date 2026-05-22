export interface PatientFeatures {
  age: number;
  blood_pressure: number;
  comorbidities: number;
}

export interface CateResponse {
  cate_estimates: number[];
  ate_estimate: number;
  ci_lower?: number[];
  ci_upper?: number[];
  magnitude?: string[];
  confidence_pct?: number[];
}

export interface PolicyResponse {
  recommended_treatment: number[];
  rationale?: string[];
  bandit_recommendation?: number[];
}

export interface DagResponse {
  nodes: string[];
  edges: { source: string; target: string }[];
}

export interface SimBothResponse {
  outcome_if_withhold: number[];
  outcome_if_treat: number[];
  treatment_effect: number[];
}

export interface ExplainResponse {
  shap_values: number[][];
  feature_labels?: string[];
  feature_names: string[];
  base_value: number;
}

export interface AnalysisResults {
  cate: CateResponse;
  policy: PolicyResponse;
  dag: DagResponse | null;
  dagRef: DagResponse;
  simBoth: SimBothResponse | null;
  explain: ExplainResponse | null;
  age: number;
  bp: number;
  comorbidities: number;
}

import type {
  PatientFeatures,
  CateResponse,
  PolicyResponse,
  DagResponse,
  SimBothResponse,
  ExplainResponse,
} from "./types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

async function callApi<T>(
  method: "GET" | "POST",
  endpoint: string,
  payload?: unknown
): Promise<T | null> {
  try {
    const res = await fetch(`${BACKEND_URL}${endpoint}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: payload !== undefined ? JSON.stringify(payload) : undefined,
      signal: AbortSignal.timeout(60_000),
    });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

function wrap(f: PatientFeatures) {
  return {
    features: [
      {
        age: f.age,
        blood_pressure: f.blood_pressure,
        comorbidities: f.comorbidities,
      },
    ],
  };
}

export const predictCate = (f: PatientFeatures) =>
  callApi<CateResponse>("POST", "/predict_cate", wrap(f));

export const policyDecision = (f: PatientFeatures) =>
  callApi<PolicyResponse>("POST", "/policy_decision", wrap(f));

export const fetchDag = () => callApi<DagResponse>("GET", "/dag");

export const fetchDagReference = () =>
  callApi<DagResponse>("GET", "/dag/reference");

export const simulateBoth = (f: PatientFeatures) =>
  callApi<SimBothResponse>("POST", "/simulate_both", wrap(f));

export const explainPrediction = (f: PatientFeatures) =>
  callApi<ExplainResponse>("POST", "/explain_prediction", wrap(f));

export async function simulateBothFallback(
  f: PatientFeatures,
  cateVal: number
): Promise<SimBothResponse | null> {
  const [r0, r1] = await Promise.all([
    callApi<{ expected_outcomes: number[] }>("POST", "/simulate_intervention", {
      ...wrap(f),
      treatment_value: 0,
    }),
    callApi<{ expected_outcomes: number[] }>("POST", "/simulate_intervention", {
      ...wrap(f),
      treatment_value: 1,
    }),
  ]);
  if (!r0 || !r1) return null;
  return {
    outcome_if_withhold: r0.expected_outcomes,
    outcome_if_treat: r1.expected_outcomes,
    treatment_effect: [cateVal],
  };
}

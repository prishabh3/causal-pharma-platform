"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import ClinicalSummary from "@/components/ClinicalSummary";
import EmptyState from "@/components/EmptyState";
import MessageBox from "@/components/MessageBox";
import RecommendationView from "@/components/views/RecommendationView";
import WhyView from "@/components/views/WhyView";
import WhatIfView from "@/components/views/WhatIfView";
import DetailsView from "@/components/views/DetailsView";
import {
  predictCate,
  policyDecision,
  fetchDag,
  fetchDagReference,
  simulateBoth,
  simulateBothFallback,
  explainPrediction,
} from "@/lib/api";
import { EXAMPLE_PATIENTS, MAGNITUDE_LABELS, REFERENCE_DAG_FALLBACK } from "@/lib/content";
import type { AnalysisResults } from "@/lib/types";

type View = "recommendation" | "why" | "whatif" | "details";

const VIEWS: { id: View; label: string }[] = [
  { id: "recommendation", label: "① Recommendation" },
  { id: "why", label: "② Why?" },
  { id: "whatif", label: "③ What-if" },
  { id: "details", label: "④ Details" },
];

export default function Home() {
  const [age, setAge] = useState(65);
  const [bp, setBp] = useState(120);
  const [comorbidities, setComorbidities] = useState(2);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [activeView, setActiveView] = useState<View>("recommendation");

  function handlePresetSelect(preset: string) {
    const p = EXAMPLE_PATIENTS[preset];
    if (p) {
      setAge(p.age);
      setBp(p.blood_pressure);
      setComorbidities(p.comorbidities);
    }
  }

  async function handleRunAnalysis() {
    setLoading(true);
    setErrors([]);

    const features = { age, blood_pressure: bp, comorbidities };
    const errs: string[] = [];

    const [cate, policy, dag, dagRefRaw, simBothRaw, explain] =
      await Promise.all([
        predictCate(features),
        policyDecision(features),
        fetchDag(),
        fetchDagReference(),
        simulateBoth(features),
        explainPrediction(features),
      ]);

    if (!cate) errs.push("`/predict_cate` — is the backend running?");
    if (!policy) errs.push("`/policy_decision`");

    const cateVal = cate?.cate_estimates?.[0] ?? 0;

    let simResult = simBothRaw;
    if (!simResult) {
      simResult = await simulateBothFallback(features, cateVal);
      if (!simResult) errs.push("`/simulate_both` (restart backend to enable)");
    }
    if (!explain) errs.push("`/explain_prediction` (restart backend for SHAP)");

    const dagRef = dagRefRaw ?? REFERENCE_DAG_FALLBACK;

    setErrors(errs);

    if (cate && policy) {
      setResults({
        cate,
        policy,
        dag: dag ?? null,
        dagRef,
        simBoth: simResult,
        explain: explain ?? null,
        age,
        bp,
        comorbidities,
      });
      setActiveView("recommendation");
    }

    setLoading(false);
  }

  const R = results;
  const hasResults = !!(R?.cate && R?.policy);

  const cateVal = R?.cate.cate_estimates[0] ?? 0;
  const ateVal = R?.cate.ate_estimate ?? 0;
  const margin = Math.max(0.3, Math.abs(cateVal) * 0.15);
  const ciLo = R?.cate.ci_lower?.[0] ?? cateVal - margin;
  const ciHi = R?.cate.ci_upper?.[0] ?? cateVal + margin;
  const magnitude =
    R?.cate.magnitude?.[0] ?? (Math.abs(cateVal) >= 2 ? "moderate" : "small");
  const rec = R?.policy.recommended_treatment[0] ?? 0;
  const recLabel = rec === 1 ? "Treat (T=1)" : "Withhold (T=0)";

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        age={age}
        bp={bp}
        comorbidities={comorbidities}
        loading={loading}
        onAgeChange={setAge}
        onBpChange={setBp}
        onComorbiditiesChange={setComorbidities}
        onPresetSelect={handlePresetSelect}
        onRunAnalysis={handleRunAnalysis}
      />

      <main className="flex-1 overflow-y-auto px-8 py-6">
        {/* Header */}
        <div className="pb-5 mb-5 border-b border-slate-200">
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Causal Pharma Intelligence
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Understand treatment effects — step by step
          </p>
        </div>

        {/* Errors (partial failure) */}
        {errors.length > 0 && (
          <MessageBox variant={hasResults ? "warn" : "error"}>
            {hasResults ? (
              <>
                Some features need a backend restart:{" "}
                <code>docker-compose restart backend</code>
                <ul className="mt-2 list-disc list-inside">
                  {errors.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                Analysis failed. Start the stack with{" "}
                <code>docker-compose up --build</code>, wait until{" "}
                <code>
                  {process.env.NEXT_PUBLIC_BACKEND_URL ||
                    "http://localhost:8000"}
                  /health
                </code>{" "}
                returns OK, then run again.
                <ul className="mt-2 list-disc list-inside">
                  {errors.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </>
            )}
          </MessageBox>
        )}

        {/* Loading spinner */}
        {loading && (
          <p className="text-sm text-slate-500 animate-pulse mt-4">
            Computing causal estimates and explanations (SHAP may take ~30s)…
          </p>
        )}

        {/* Results */}
        {hasResults && R ? (
          <>
            <ClinicalSummary
              age={R.age}
              bp={R.bp}
              comorbidities={R.comorbidities}
              cateVal={cateVal}
              ateVal={ateVal}
              recLabel={recLabel}
              magnitude={magnitude}
              ciLo={ciLo}
              ciHi={ciHi}
            />

            {/* View tabs */}
            <div className="flex gap-1 mb-5 border-b border-slate-200">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setActiveView(v.id)}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeView === v.id
                      ? "text-slate-900 border-b-2 border-violet-600 bg-white"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>

            {activeView === "recommendation" && (
              <RecommendationView results={R} />
            )}
            {activeView === "why" && <WhyView results={R} />}
            {activeView === "whatif" && <WhatIfView results={R} />}
            {activeView === "details" && <DetailsView results={R} />}
          </>
        ) : (
          !loading && <EmptyState age={age} bp={bp} comorbidities={comorbidities} />
        )}
      </main>
    </div>
  );
}

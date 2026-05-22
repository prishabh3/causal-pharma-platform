import MetricCard from "../MetricCard";
import MessageBox from "../MessageBox";
import { GLOSSARY, MAGNITUDE_LABELS } from "@/lib/content";
import type { AnalysisResults } from "@/lib/types";

interface Props {
  results: AnalysisResults;
}

export default function RecommendationView({ results }: Props) {
  const { cate, policy } = results;
  const cateVal = cate.cate_estimates[0];
  const ateVal = cate.ate_estimate;
  const margin = Math.max(0.3, Math.abs(cateVal) * 0.15);
  const ciLo = cate.ci_lower?.[0] ?? cateVal - margin;
  const ciHi = cate.ci_upper?.[0] ?? cateVal + margin;
  const magnitude = cate.magnitude?.[0] ?? (Math.abs(cateVal) >= 2 ? "moderate" : "small");
  const confidence = cate.confidence_pct?.[0] ?? 80;

  const rec = policy.recommended_treatment[0];
  const recLabel = rec === 1 ? "Treat (T=1)" : "Withhold (T=0)";
  const rationale =
    policy.rationale?.[0] ??
    `Estimated treatment effect is ${cateVal >= 0 ? "+" : ""}${cateVal.toFixed(2)} outcome units for this patient.`;
  const banditRec = policy.bandit_recommendation?.[0] ?? rec;
  const banditLabel = banditRec === 1 ? "Treat (T=1)" : "Withhold (T=0)";

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="ATE (avg. effect)"
          value={`${ateVal >= 0 ? "+" : ""}${ateVal.toFixed(3)}`}
          help="Average benefit across similar patients in training data"
        />
        <MetricCard
          label="CATE (this patient)"
          value={`${cateVal >= 0 ? "+" : ""}${cateVal.toFixed(3)}`}
          delta={`95% CI [${ciLo >= 0 ? "+" : ""}${ciLo.toFixed(2)}, ${ciHi >= 0 ? "+" : ""}${ciHi.toFixed(2)}]`}
          help="Expected benefit for this specific patient"
        />
        <MetricCard
          label="Policy"
          value={recLabel}
          help="Recommended treatment based on CATE sign"
        />
        <MetricCard
          label="Impact"
          value={MAGNITUDE_LABELS[magnitude] ?? magnitude}
          delta={`~${confidence}% confidence`}
        />
      </div>

      <MessageBox variant="info">{rationale}</MessageBox>

      {banditRec !== rec ? (
        <MessageBox variant="warn">
          LinUCB bandit (exploratory) suggests <strong>{banditLabel}</strong> —
          may differ when exploring trade-offs.
        </MessageBox>
      ) : (
        <p className="text-xs text-slate-500">
          LinUCB bandit agrees with the CATE-based policy ({banditLabel}).
        </p>
      )}

      <details className="border border-slate-200 rounded-xl">
        <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 rounded-xl select-none">
          What do these numbers mean?
        </summary>
        <div className="px-4 pb-4 pt-2 flex flex-col gap-2">
          {Object.entries(GLOSSARY).map(([term, def]) => (
            <p key={term} className="text-sm text-slate-700">
              <strong>{term}</strong> — {def}
            </p>
          ))}
        </div>
      </details>
    </div>
  );
}

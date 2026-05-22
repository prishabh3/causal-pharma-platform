import MetricCard from "../MetricCard";
import { MAGNITUDE_LABELS } from "@/lib/content";
import type { AnalysisResults } from "@/lib/types";

interface Props {
  results: AnalysisResults;
}

export default function DetailsView({ results }: Props) {
  const { cate, policy, simBoth, age, bp, comorbidities } = results;
  const cateVal = cate.cate_estimates[0];
  const ateVal = cate.ate_estimate;
  const magnitude = cate.magnitude?.[0] ?? (Math.abs(cateVal) >= 2 ? "moderate" : "small");
  const confidence = cate.confidence_pct?.[0] ?? 80;
  const rec = policy.recommended_treatment[0];
  const recLabel = rec === 1 ? "Treat (T=1)" : "Withhold (T=0)";

  const patientId = `P-${String(Math.round(age)).padStart(3, "0")}${String(Math.round(bp)).padStart(3, "0")}${String(comorbidities).padStart(2, "0")}`;

  const y0 = simBoth?.outcome_if_withhold?.[0];
  const y1 = simBoth?.outcome_if_treat?.[0];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h3 className="text-sm font-semibold text-slate-900 mb-3">
          Patient summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="Patient ID" value={patientId} />
          <MetricCard label="Age" value={`${Math.round(age)} yrs`} />
          <MetricCard label="Blood pressure" value={`${Math.round(bp)} mmHg`} />
          <MetricCard label="Comorbidities" value={String(comorbidities)} />
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-slate-900 mb-3">
          Treatment analysis
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="ATE"
            value={`${ateVal >= 0 ? "+" : ""}${ateVal.toFixed(3)}`}
          />
          <MetricCard
            label="CATE"
            value={`${cateVal >= 0 ? "+" : ""}${cateVal.toFixed(3)}`}
          />
          <MetricCard label="Policy" value={recLabel} />
          <MetricCard label="Confidence" value={`${confidence}%`} />
        </div>
      </div>

      {y0 !== undefined && y1 !== undefined && (
        <div>
          <h3 className="text-sm font-semibold text-slate-900 mb-3">
            Outcome comparison
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-slate-200 rounded-xl overflow-hidden">
              <thead className="bg-slate-50">
                <tr>
                  {["Scenario", "Expected outcome", "vs control"].map((h) => (
                    <th
                      key={h}
                      className="text-left px-4 py-2 text-xs font-semibold text-slate-600 uppercase tracking-wide"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr>
                  <td className="px-4 py-2 text-slate-700">Control (T=0)</td>
                  <td className="px-4 py-2 font-mono text-slate-900">{y0.toFixed(1)}</td>
                  <td className="px-4 py-2 text-slate-400">—</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-slate-700">Treated (T=1)</td>
                  <td className="px-4 py-2 font-mono text-slate-900">{y1.toFixed(1)}</td>
                  <td
                    className="px-4 py-2 font-mono font-semibold"
                    style={{ color: y1 - y0 >= 0 ? "#059669" : "#dc2626" }}
                  >
                    {y1 - y0 >= 0 ? "+" : ""}
                    {(y1 - y0).toFixed(1)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div>
        <h3 className="text-sm font-semibold text-slate-900 mb-2">
          Effect magnitude
        </h3>
        <p className="text-sm text-slate-700">
          {MAGNITUDE_LABELS[magnitude] ?? magnitude}
        </p>
      </div>
    </div>
  );
}

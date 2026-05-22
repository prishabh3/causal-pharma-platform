import MetricCard from "../MetricCard";
import HorizontalBarChart from "../HorizontalBarChart";
import MessageBox from "../MessageBox";
import type { AnalysisResults } from "@/lib/types";

interface Props {
  results: AnalysisResults;
}

export default function WhatIfView({ results }: Props) {
  const { simBoth, cate } = results;
  const cateVal = cate.cate_estimates[0];

  const y0 = simBoth?.outcome_if_withhold?.[0];
  const y1 = simBoth?.outcome_if_treat?.[0];

  if (y0 === undefined || y1 === undefined) {
    return (
      <MessageBox variant="warn">Counterfactual outcomes unavailable.</MessageBox>
    );
  }

  const diff = y1 - y0;
  const sign = diff > 0 ? "more" : "less";

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="If we withhold (T=0)" value={y0.toFixed(1)} />
        <MetricCard label="If we treat (T=1)" value={y1.toFixed(1)} />
        <MetricCard
          label="Difference"
          value={`${diff >= 0 ? "+" : ""}${diff.toFixed(1)}`}
        />
      </div>

      <HorizontalBarChart
        rows={[
          ["Withhold (T=0)", y0],
          ["Treat (T=1)", y1],
        ]}
        title="Expected outcome comparison"
      />

      <MessageBox variant="info">
        <strong>Treating vs withholding:</strong> expected outcome is{" "}
        <strong>
          {Math.abs(diff).toFixed(1)} units {sign}
        </strong>{" "}
        with treatment (individual effect ≈{" "}
        {cateVal >= 0 ? "+" : ""}
        {cateVal.toFixed(2)}).
      </MessageBox>
    </div>
  );
}

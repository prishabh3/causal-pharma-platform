import DagMermaid from "../DagMermaid";
import HorizontalBarChart from "../HorizontalBarChart";
import MessageBox from "../MessageBox";
import type { AnalysisResults } from "@/lib/types";

interface Props {
  results: AnalysisResults;
}

export default function WhyView({ results }: Props) {
  const { dag, dagRef, explain } = results;

  const shapRows: [string, number][] | null =
    explain?.shap_values?.[0]
      ? (explain.feature_labels ?? explain.feature_names).map(
          (label, i) => [label, explain.shap_values[0][i]] as [string, number]
        )
      : null;

  return (
    <div className="flex flex-col gap-6">
      <p className="text-xs text-slate-500">
        Causal links from data (left) vs clinical reference (right).
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {dag && (
          <DagMermaid
            nodes={dag.nodes}
            edges={dag.edges}
            title="Learned from data"
            subtitle={
              dag.edges.length > 0
                ? `${dag.edges.length} edge(s) from NOTEARS`
                : "No strong edges at current threshold"
            }
          />
        )}
        <DagMermaid
          nodes={dagRef.nodes}
          edges={dagRef.edges}
          title="Clinical reference"
          subtitle="Expected relationships from domain knowledge"
        />
      </div>

      <div>
        <h3 className="text-sm font-semibold text-slate-900 mb-1">
          Feature drivers (SHAP)
        </h3>
        {shapRows ? (
          <>
            <HorizontalBarChart
              rows={shapRows}
              title={`Why this CATE? (baseline = ${explain!.base_value >= 0 ? "+" : ""}${explain!.base_value.toFixed(3)})`}
            />
            <p className="text-xs text-slate-500 mt-1">
              Green = pushes treatment effect up · Red = pushes it down
            </p>
          </>
        ) : (
          <MessageBox variant="warn">
            SHAP explanation unavailable. Restart backend:{" "}
            <code>docker-compose restart backend</code>
          </MessageBox>
        )}
      </div>
    </div>
  );
}

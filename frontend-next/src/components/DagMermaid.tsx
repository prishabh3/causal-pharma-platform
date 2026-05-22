"use client";

import { useEffect, useRef } from "react";

interface Edge {
  source: string;
  target: string;
}

interface Props {
  nodes: string[];
  edges: Edge[];
  title: string;
  subtitle?: string;
}

function toLabel(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function toId(name: string) {
  return name.replace(/[\s\-]/g, "_");
}

let mermaidIdCounter = 0;

export default function DagMermaid({ nodes, edges, title, subtitle }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(`dag_${mermaidIdCounter++}`);

  useEffect(() => {
    if (!containerRef.current) return;

    const lines = ["flowchart LR"];
    const linked = new Set<string>();
    for (const e of edges) {
      const s = toId(e.source);
      const t = toId(e.target);
      lines.push(`    ${s}["${toLabel(e.source)}"] --> ${t}["${toLabel(e.target)}"]`);
      linked.add(s);
      linked.add(t);
    }
    for (const n of nodes) {
      const mid = toId(n);
      if (!linked.has(mid)) {
        lines.push(`    ${mid}["${toLabel(n)}"]`);
      }
    }
    const code = lines.join("\n");

    import("mermaid").then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        themeVariables: {
          primaryColor: "#eff6ff",
          primaryBorderColor: "#3b82f6",
          edgeLabelBackground: "#ffffff",
          fontFamily: "Inter, sans-serif",
        },
      });
      mermaid
        .render(idRef.current, code)
        .then(({ svg }) => {
          if (containerRef.current) {
            containerRef.current.innerHTML = svg;
          }
        })
        .catch(() => {
          if (containerRef.current) {
            containerRef.current.innerHTML =
              '<p class="text-xs text-slate-400">DAG render failed</p>';
          }
        });
    });
  }, [nodes, edges]);

  return (
    <div>
      <p className="text-sm font-semibold text-slate-900">{title}</p>
      {subtitle && <p className="text-xs text-slate-500 mb-2">{subtitle}</p>}
      <div
        ref={containerRef}
        className="flex justify-center items-center min-h-48 text-xs text-slate-400"
      >
        Rendering…
      </div>
    </div>
  );
}

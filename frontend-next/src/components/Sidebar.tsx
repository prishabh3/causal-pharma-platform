"use client";

import { EXAMPLE_PATIENTS, GLOSSARY, ABOUT_ANALYSIS } from "@/lib/content";

interface Props {
  age: number;
  bp: number;
  comorbidities: number;
  loading: boolean;
  onAgeChange: (v: number) => void;
  onBpChange: (v: number) => void;
  onComorbiditiesChange: (v: number) => void;
  onPresetSelect: (preset: string) => void;
  onRunAnalysis: () => void;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mt-4 mb-1.5">
      {children}
    </p>
  );
}

function Divider() {
  return <hr className="border-slate-100 my-3" />;
}

export default function Sidebar({
  age,
  bp,
  comorbidities,
  loading,
  onAgeChange,
  onBpChange,
  onComorbiditiesChange,
  onPresetSelect,
  onRunAnalysis,
}: Props) {
  return (
    <aside className="w-72 shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-y-auto p-4">
      <SectionLabel>Example patient</SectionLabel>
      <select
        className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-800 bg-white focus:outline-none focus:ring-2 focus:ring-slate-300"
        defaultValue="Custom (use sliders)"
        onChange={(e) => onPresetSelect(e.target.value)}
      >
        {Object.keys(EXAMPLE_PATIENTS).map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>

      <SectionLabel>Patient parameters</SectionLabel>

      <label className="text-xs font-medium text-slate-600 mb-1">
        Age — {Math.round(age)} yrs
      </label>
      <input
        type="range"
        min={18}
        max={95}
        value={age}
        onChange={(e) => onAgeChange(Number(e.target.value))}
        className="w-full accent-slate-900 mb-3"
      />

      <label className="text-xs font-medium text-slate-600 mb-1">
        Blood pressure — {Math.round(bp)} mmHg
      </label>
      <input
        type="range"
        min={80}
        max={200}
        value={bp}
        onChange={(e) => onBpChange(Number(e.target.value))}
        className="w-full accent-slate-900 mb-3"
      />

      <label className="text-xs font-medium text-slate-600 mb-1">
        Comorbidities
      </label>
      <div className="flex items-center gap-2 mb-3">
        <button
          className="w-8 h-8 rounded-lg border border-slate-200 text-slate-700 font-bold hover:bg-slate-50"
          onClick={() => onComorbiditiesChange(Math.max(0, comorbidities - 1))}
        >
          −
        </button>
        <span className="flex-1 text-center font-mono text-sm font-semibold text-slate-900">
          {comorbidities}
        </span>
        <button
          className="w-8 h-8 rounded-lg border border-slate-200 text-slate-700 font-bold hover:bg-slate-50"
          onClick={() => onComorbiditiesChange(Math.min(10, comorbidities + 1))}
        >
          +
        </button>
      </div>

      <Divider />

      <details className="mb-2">
        <summary className="cursor-pointer text-sm font-semibold text-slate-700 select-none hover:text-slate-900">
          Glossary
        </summary>
        <div className="mt-2 flex flex-col gap-2">
          {Object.entries(GLOSSARY).map(([term, def]) => (
            <p key={term} className="text-xs text-slate-600 leading-relaxed">
              <strong className="text-slate-800">{term}</strong> — {def}
            </p>
          ))}
        </div>
      </details>

      <details className="mb-2">
        <summary className="cursor-pointer text-sm font-semibold text-slate-700 select-none hover:text-slate-900">
          About this analysis
        </summary>
        <div className="mt-2 text-xs text-slate-600 leading-relaxed whitespace-pre-line">
          {ABOUT_ANALYSIS.trim()
            .split("\n")
            .filter((l) => !l.startsWith("|---"))
            .map((line, i) => (
              <p key={i} className={line.startsWith("|") ? "py-0.5" : "py-1"}>
                {line
                  .replace(/^\| /, "")
                  .replace(/ \|$/, "")
                  .replace(/\*\*(.*?)\*\*/g, "$1")
                  .replace(/ \| /g, " · ")}
              </p>
            ))}
        </div>
      </details>

      <Divider />

      <button
        className="w-full bg-slate-900 text-white rounded-lg py-2.5 text-sm font-semibold hover:bg-slate-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={onRunAnalysis}
        disabled={loading}
      >
        {loading ? "Computing…" : "Run Analysis"}
      </button>
    </aside>
  );
}

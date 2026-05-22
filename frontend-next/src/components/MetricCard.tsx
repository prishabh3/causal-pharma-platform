interface Props {
  label: string;
  value: string;
  delta?: string;
  help?: string;
}

export default function MetricCard({ label, value, delta, help }: Props) {
  return (
    <div
      className="bg-white border border-slate-200 rounded-xl px-4 py-3 flex flex-col gap-1"
      title={help}
    >
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <span className="font-mono text-2xl font-bold text-slate-900 leading-tight">
        {value}
      </span>
      {delta && (
        <span className="text-xs text-slate-500">{delta}</span>
      )}
    </div>
  );
}

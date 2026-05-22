interface Props {
  rows: [string, number][];
  title?: string;
}

export default function HorizontalBarChart({ rows, title }: Props) {
  const maxAbs = Math.max(...rows.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 my-2">
      {title && (
        <p className="text-sm font-semibold text-slate-900 mb-3">{title}</p>
      )}
      <div className="flex flex-col gap-2.5">
        {rows.map(([label, value]) => {
          const pct = Math.max(4, Math.round((100 * Math.abs(value)) / maxAbs));
          const color = value >= 0 ? "#059669" : "#dc2626";
          return (
            <div key={label} className="flex items-center gap-3">
              <span className="w-32 shrink-0 text-sm font-semibold text-slate-900 truncate">
                {label}
              </span>
              <div className="flex-1 h-6 bg-slate-100 rounded-md overflow-hidden">
                <div
                  className="h-full rounded-md"
                  style={{ width: `${pct}%`, background: color, minWidth: 4 }}
                />
              </div>
              <span
                className="w-18 shrink-0 text-right text-sm font-bold font-mono"
                style={{ color, minWidth: 72 }}
              >
                {value >= 0 ? "+" : ""}
                {value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

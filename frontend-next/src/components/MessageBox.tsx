interface Props {
  children: React.ReactNode;
  variant?: "info" | "ok" | "warn" | "error";
}

const styles = {
  info: "bg-blue-50 border-blue-500 text-slate-900",
  ok: "bg-emerald-50 border-emerald-600 text-slate-900",
  warn: "bg-amber-50 border-amber-500 text-slate-900",
  error: "bg-red-50 border-red-500 text-slate-900",
};

export default function MessageBox({ children, variant = "info" }: Props) {
  return (
    <div
      className={`border-l-4 rounded-lg px-4 py-3 my-3 text-sm leading-relaxed ${styles[variant]}`}
    >
      {children}
    </div>
  );
}

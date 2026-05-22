import MessageBox from "./MessageBox";

interface Props {
  age: number;
  bp: number;
  comorbidities: number;
}

export default function EmptyState({ age, bp, comorbidities }: Props) {
  return (
    <div>
      <MessageBox variant="info">
        <strong>Ready for analysis</strong> — Choose an{" "}
        <strong>example patient</strong> in the sidebar (or adjust sliders),
        then click <strong>Run Analysis</strong>.
      </MessageBox>
      <p className="text-sm text-slate-700 mt-2">
        <strong>Current patient:</strong> age <strong>{Math.round(age)}</strong>
        , BP <strong>{Math.round(bp)}</strong>, comorbidities{" "}
        <strong>{comorbidities}</strong>
      </p>
    </div>
  );
}

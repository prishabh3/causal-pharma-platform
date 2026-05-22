import MessageBox from "./MessageBox";
import { MAGNITUDE_LABELS } from "@/lib/content";

interface Props {
  age: number;
  bp: number;
  comorbidities: number;
  cateVal: number;
  ateVal: number;
  recLabel: string;
  magnitude: string;
  ciLo: number;
  ciHi: number;
}

export default function ClinicalSummary({
  age,
  bp,
  comorbidities,
  cateVal,
  ateVal,
  recLabel,
  magnitude,
  ciLo,
  ciHi,
}: Props) {
  const direction =
    cateVal > 0 ? "improve" : cateVal < 0 ? "worsen" : "not materially change";
  const magText = MAGNITUDE_LABELS[magnitude] ?? "Unknown impact";
  const comorbWord = comorbidities === 1 ? "comorbidity" : "comorbidities";

  return (
    <MessageBox variant="ok">
      <strong>Clinical summary</strong> — For a{" "}
      <strong>{Math.round(age)}-year-old</strong> patient with{" "}
      <strong>BP {Math.round(bp)} mmHg</strong> and{" "}
      <strong>
        {comorbidities} {comorbWord}
      </strong>
      , treatment is estimated to <strong>{direction}</strong> the outcome by{" "}
      <strong>{Math.abs(cateVal).toFixed(2)} units</strong> (
      {magText.toLowerCase()}). 95% interval:{" "}
      <strong>
        [{ciLo >= 0 ? "+" : ""}
        {ciLo.toFixed(2)}, {ciHi >= 0 ? "+" : ""}
        {ciHi.toFixed(2)}]
      </strong>
      . <strong>Recommended: {recLabel}</strong>. ATE:{" "}
      <strong>
        {ateVal >= 0 ? "+" : ""}
        {ateVal.toFixed(2)}
      </strong>
      .
    </MessageBox>
  );
}

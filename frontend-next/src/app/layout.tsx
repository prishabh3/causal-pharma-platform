import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Causal Pharma Intelligence",
  description: "Understand treatment effects — step by step",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}

import React from "react";
import { useGsapStagger } from "../hooks/useGsapStagger";
import { C, display } from "../theme";

type Props = {
  text: string;
  size: number;
  start?: number; // seconds into the Sequence
  color?: string;
  highlight?: string[]; // words drawn in teal
  danger?: string[]; // words drawn in red
  align?: "left" | "center";
  maxWidth?: number;
  weight?: number;
};

/** Kinetic headline: words rise in on a GSAP stagger (deterministic, frame-seeked). */
export const Kinetic: React.FC<Props> = ({ text, size, start = 0, color = C.ink, highlight = [], danger = [], align = "left", maxWidth, weight = 800 }) => {
  const words = text.split(" ");
  const p = useGsapStagger(words.length, { start, each: 0.07, duration: 0.6, ease: "back.out(1.6)" });
  const clean = (w: string) => w.replace(/[.,!?—:]/g, "");
  return (
    <div style={{ fontFamily: display, fontWeight: weight, fontSize: size, lineHeight: 1.04, letterSpacing: "-0.035em",
      color, textAlign: align, maxWidth, display: "flex", flexWrap: "wrap", gap: `0 ${size * 0.24}px`,
      justifyContent: align === "center" ? "center" : "flex-start" }}>
      {words.map((w, i) => (
        <span key={i} style={{ display: "inline-block", opacity: p[i], transform: `translateY(${(1 - p[i]) * size * 0.5}px)`,
          color: highlight.includes(clean(w)) ? C.teal : danger.includes(clean(w)) ? C.red : undefined }}>
          {w}
        </span>
      ))}
    </div>
  );
};

import React from "react";
import { AbsoluteFill } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { useLayout } from "../components/Layout";
import { useReveal } from "../hooks/useSpringIn";
import { body, C, display } from "../theme";

const LINES: [string, string][] = [
  ["Every ticket", "billed by the token"],
  ["Every customer email", "on someone else's server"],
  ["Every outage", "your agent stops working"],
];

const Line: React.FC<{ i: number; a: string; b: string; size: number }> = ({ i, a, b, size }) => {
  const s = useReveal(i * 16, 50);
  return (
    <div style={{ ...s, display: "flex", flexDirection: "column", gap: 6, borderLeft: `6px solid ${C.red}`, paddingLeft: 28 }}>
      <div style={{ fontFamily: display, fontWeight: 800, fontSize: size, color: C.ink, letterSpacing: "-0.03em" }}>{a}</div>
      <div style={{ fontFamily: body, fontSize: size * 0.55, color: "#FFB3B0" }}>→ {b}</div>
    </div>
  );
};

/** 3–7s · agitate: three costs, then the question that turns the film. */
export const Agitate: React.FC = () => {
  const { vertical } = useLayout();
  const size = vertical ? 76 : 70;
  return (
    <AbsoluteFill style={{ padding: vertical ? "330px 90px" : "150px 140px", display: "flex", flexDirection: vertical ? "column" : "row",
      gap: vertical ? 70 : 120, alignItems: vertical ? "stretch" : "center" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: vertical ? 54 : 48, flex: vertical ? undefined : 1 }}>
        {LINES.map(([a, b], i) => <Line key={i} i={i} a={a} b={b} size={size} />)}
      </div>
      <div style={{ flex: vertical ? undefined : 1, marginTop: vertical ? 40 : 0 }}>
        <Kinetic text="Does sorting tickets really need a frontier model?" size={vertical ? 84 : 84} start={2.0} highlight={["really"]} />
      </div>
    </AbsoluteFill>
  );
};

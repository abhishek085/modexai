import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { useLayout } from "../components/Layout";
import { useGsapStagger } from "../hooks/useGsapStagger";
import { body, C, display } from "../theme";
import type { LaunchProps } from "../types";

const CHIPS = ["0 tokens billed", "0 bytes sent", "works offline"];

/** 18.5–22.5s · the payoff: work flows to the local hire; the meter that mattered reads zero. */
export const Delegate: React.FC<LaunchProps> = ({ hire, hardware }) => {
  const frame = useCurrentFrame();
  const { vertical } = useLayout();
  const n = Math.floor(interpolate(frame, [8, 80], [0, 300], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const chips = useGsapStagger(CHIPS.length, { start: 1.2, each: 0.18, duration: 0.6, ease: "back.out(2)" });
  return (
    <AbsoluteFill style={{ padding: vertical ? "300px 90px" : "160px 140px", display: "flex", flexDirection: vertical ? "column" : "row",
      gap: vertical ? 60 : 120, alignItems: vertical ? "flex-start" : "center" }}>
      <div style={{ flex: vertical ? undefined : 1 }}>
        <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 300 : 280, lineHeight: 0.9, color: C.teal,
          letterSpacing: "-0.06em", fontVariantNumeric: "tabular-nums" }}>{n}</div>
        <div style={{ fontFamily: body, fontSize: vertical ? 44 : 42, color: C.ink, marginTop: 16 }}>
          new tickets handled by <b>{hire.tag}</b>
        </div>
        <div style={{ fontFamily: body, fontSize: vertical ? 32 : 30, color: C.muted, marginTop: 8 }}>on {hardware}</div>
      </div>
      <div style={{ flex: vertical ? undefined : 1, display: "flex", flexDirection: "column", gap: 28 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
          {CHIPS.map((c, i) => (
            <div key={c} style={{ opacity: chips[i], transform: `scale(${0.6 + 0.4 * chips[i]})`, fontFamily: body, fontWeight: 600,
              fontSize: vertical ? 40 : 38, color: C.ink, border: `3px solid ${C.teal}`, borderRadius: 999, padding: "16px 30px" }}>{c}</div>
          ))}
        </div>
        <div style={{ marginTop: 30 }}>
          <Kinetic text="The work stays home." size={vertical ? 96 : 100} start={1.9} highlight={["home"]} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

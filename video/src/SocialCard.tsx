import React from "react";
import { AbsoluteFill } from "remotion";
import { Backdrop, LogoMark } from "./components/Layout";
import { body, C, display, mono } from "./theme";
import type { LaunchProps } from "./types";

/** 1280×640 GitHub social preview, built from the same design system and real data. */
export const SocialCard: React.FC<LaunchProps> = ({ race, hire, hardware, bar }) => (
  <AbsoluteFill>
    <Backdrop />
    <AbsoluteFill style={{ padding: "64px 72px", display: "flex", flexDirection: "row", gap: 56 }}>
      <div style={{ flex: 1.1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <LogoMark size={72} />
          <div style={{ fontFamily: display, fontWeight: 800, fontSize: 64, letterSpacing: "-0.05em", color: C.white }}>
            modex<span style={{ color: C.teal }}>ai</span>
          </div>
        </div>
        <div style={{ fontFamily: display, fontWeight: 800, fontSize: 68, lineHeight: 1.02, letterSpacing: "-0.04em", color: C.ink }}>
          What runs locally, measured.
        </div>
        <div style={{ fontFamily: body, fontSize: 27, color: C.muted, lineHeight: 1.4 }}>
          Your agent hires a small local model for the jobs it can prove it does, and shares the recipe. No weights, no data.
        </div>
        <div style={{ fontFamily: body, fontSize: 22, color: C.muted }}>MCP server · CLI · open exchange · a Nokast tool</div>
      </div>
      <div style={{ flex: 1, alignSelf: "center", background: "#0A1020", border: `2px solid ${C.line}`, borderRadius: 24, padding: "26px 28px" }}>
        <div style={{ fontFamily: mono, fontSize: 20, color: C.muted, marginBottom: 14 }}>ticket-triage · {hardware}</div>
        {race.map((r) => (
          <div key={r.tag} style={{ display: "grid", gridTemplateColumns: "170px 1fr 70px", gap: 14, alignItems: "center", padding: "12px 0",
            borderTop: `2px solid ${C.line}` }}>
            <div style={{ fontFamily: mono, fontSize: 21, color: C.ink }}>{r.tag}</div>
            <div style={{ height: 14, background: "#1C2740", borderRadius: 4, position: "relative" }}>
              <div style={{ position: "absolute", inset: 0, width: `${r.score * 100}%`, borderRadius: 4,
                background: r.score >= bar ? C.teal : r.score >= bar - 0.15 ? C.amber : "#5B6474" }} />
            </div>
            <div style={{ fontFamily: display, fontWeight: 800, fontSize: 26, textAlign: "right", color: r.score >= bar ? C.teal : C.muted }}>
              {Math.round(r.score * 100)}%
            </div>
          </div>
        ))}
        <div style={{ marginTop: 16, background: C.teal, color: "#04201D", borderRadius: 14, padding: "14px 18px", fontFamily: display,
          fontWeight: 800, fontSize: 28 }}>Hire {hire.tag} · {hire.size} GB</div>
      </div>
    </AbsoluteFill>
  </AbsoluteFill>
);

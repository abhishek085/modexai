import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { useLayout } from "../components/Layout";
import { Terminal } from "../components/Terminal";
import { useGsapStagger } from "../hooks/useGsapStagger";
import { useReveal, useSpringIn } from "../hooks/useSpringIn";
import { body, C, display, mono } from "../theme";
import type { LaunchProps } from "../types";

const CMD = "$ modexai hire ticket-triage";

/** 11–18.5s · show, don't tell: the live race board from the real CLI, with real numbers. */
export const Race: React.FC<LaunchProps> = ({ race, hire, hardware, bar }) => {
  const frame = useCurrentFrame();
  const { vertical, width } = useLayout();
  const typed = CMD.slice(0, Math.floor(interpolate(frame, [12, 40], [0, CMD.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })));
  const progress = useGsapStagger(race.length, { start: 1.5, each: 0.25, duration: 2.6, ease: "power2.inOut" });
  const termIn = useReveal(4, 60);
  const verdictIn = useSpringIn(150);
  const W = vertical ? 940 : 1180;
  const rowH = vertical ? 118 : 96;
  return (
    <AbsoluteFill style={{ alignItems: vertical ? "center" : "flex-start", padding: vertical ? "220px 70px 0" : "90px 110px 0" }}>
      <div style={{ width: vertical ? 940 : 1700, marginBottom: vertical ? 60 : 44 }}>
        <Kinetic text="It tests small models on your agent's own answers — on your machine." size={vertical ? 66 : 66} highlight={["own", "your", "machine"]} />
      </div>
      <div style={{ display: "flex", gap: 60, alignItems: "flex-start", width: vertical ? undefined : width - 220 }}>
        <div style={termIn}>
          <Terminal title="modexai — zsh" width={W}>
            <div style={{ fontFamily: mono, fontSize: vertical ? 32 : 28, color: C.ink, marginBottom: 26 }}>
              {typed}<span style={{ opacity: frame % 30 < 15 ? 1 : 0, color: C.teal }}>▌</span>
            </div>
            {race.map((r, i) => {
              const p = progress[i];
              const pass = r.score >= bar;
              const shown = Math.round(r.score * 100 * p);
              const isHire = r.tag === hire.tag && verdictIn > 0.05;
              return (
                <div key={r.tag} style={{ display: "grid", gridTemplateColumns: vertical ? "300px 1fr 120px" : "300px 1fr 120px 150px",
                  alignItems: "center", gap: 22, height: rowH, borderTop: `2px solid ${C.line}`,
                  background: isHire ? `linear-gradient(90deg, ${C.teal}2E, transparent 80%)` : undefined, padding: "0 10px", borderRadius: 10 }}>
                  <div style={{ fontFamily: mono, fontSize: vertical ? 32 : 28, color: C.ink, fontWeight: 600 }}>{r.tag}</div>
                  <div style={{ height: 22, background: "#1C2740", borderRadius: 6, position: "relative", overflow: "hidden" }}>
                    <div style={{ position: "absolute", inset: 0, width: `${r.score * 100 * p}%`, borderRadius: 6,
                      background: p < 1 ? C.blue : pass ? C.teal : r.score >= bar - 0.15 ? C.amber : "#5B6474" }} />
                    <div style={{ position: "absolute", left: `${bar * 100}%`, top: -6, bottom: -6, borderLeft: `3px dashed ${C.ink}88` }} />
                  </div>
                  <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 44 : 38, textAlign: "right",
                    color: p < 1 ? C.ink : pass ? C.teal : C.muted, fontVariantNumeric: "tabular-nums" }}>{shown}%</div>
                  {!vertical && <div style={{ fontFamily: mono, fontSize: 26, color: C.muted, textAlign: "right" }}>{Math.round(r.p50)} ms</div>}
                </div>
              );
            })}
            <div style={{ fontFamily: body, fontSize: vertical ? 26 : 24, color: C.muted, marginTop: 20 }}>
              running on {hardware} · nothing leaves it
            </div>
          </Terminal>
        </div>
      </div>
      <div style={{ position: vertical ? "relative" : "absolute", right: vertical ? undefined : 110, top: vertical ? undefined : 360,
        marginTop: vertical ? 50 : 0, width: vertical ? 940 : 420, opacity: verdictIn,
        transform: `translateY(${(1 - verdictIn) * 60}px) scale(${0.9 + 0.1 * verdictIn})` }}>
        <div style={{ background: C.teal, color: "#04201D", borderRadius: 28, padding: vertical ? "34px 40px" : "34px 34px" }}>
          <div style={{ fontFamily: body, fontWeight: 600, fontSize: 30, opacity: 0.8 }}>verdict</div>
          <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 64 : 56, letterSpacing: "-0.03em", lineHeight: 1.05, marginTop: 6 }}>
            Hire {hire.tag}
          </div>
          <div style={{ fontFamily: body, fontWeight: 500, fontSize: vertical ? 34 : 30, marginTop: 12 }}>
            {Math.round(hire.score * 100)}% · {Math.round(hire.p50)} ms · {hire.size} GB — the smallest model that clears the bar
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

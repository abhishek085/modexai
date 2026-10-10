import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { useLayout } from "../components/Layout";
import { useGsapStagger } from "../hooks/useGsapStagger";
import { useSpringIn } from "../hooks/useSpringIn";
import { body, C, display, mono } from "../theme";
import type { LaunchProps } from "../types";

// Illustrative machines — the point is that anyone can reproduce a recipe on their own hardware.
const NODES = ["RTX 4090 · 24 GB", "MacBook Air M2 · 16 GB", "Raspberry Pi 5 · 8 GB", "Framework · Ryzen 7", "Mac mini M4 · 32 GB", "Jetson Orin · 16 GB"];

/** 22.5–27.5s · the network effect: the finding leaves as a recipe (no weights, no data) and others try it. */
export const Share: React.FC<LaunchProps> = ({ hire, hardware, suite }) => {
  const frame = useCurrentFrame();
  const { vertical, width, height } = useLayout();
  const card = useSpringIn(14);
  const nodes = useGsapStagger(NODES.length, { start: 1.6, each: 0.16, duration: 0.7, ease: "back.out(1.8)" });
  const draw = interpolate(frame, [50, 100], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const cx = vertical ? width / 2 : width * 0.36;
  const cy = vertical ? 900 : height * 0.56;
  const pos = NODES.map((_, i) => {
    if (vertical) {
      const col = i % 2, row = Math.floor(i / 2);
      return { x: col === 0 ? 270 : width - 270, y: 1240 + row * 150 };
    }
    const a = (-60 + i * 24) * (Math.PI / 180);
    return { x: width * 0.73 + Math.cos(a) * 60, y: height * 0.27 + i * 110 };
  });
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: vertical ? 90 : 120, top: vertical ? 230 : 90, width: vertical ? 900 : 1600 }}>
        <Kinetic text="Then it shares a recipe — not weights, not data." size={vertical ? 70 : 72} highlight={["recipe"]} />
      </div>
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        {pos.map((p, i) => (
          <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke={C.teal} strokeWidth="3" strokeOpacity={0.6}
            strokeDasharray="1200" strokeDashoffset={1200 * (1 - draw)} />
        ))}
      </svg>
      <div style={{ position: "absolute", left: cx - (vertical ? 400 : 380), top: cy - (vertical ? 250 : 230), width: vertical ? 800 : 760,
        opacity: card, transform: `translateY(${(1 - card) * 80}px) rotate(${(1 - card) * -4}deg)`,
        background: C.night2, border: `3px solid ${C.teal}`, borderRadius: 28, padding: "30px 36px",
        boxShadow: `0 30px 90px ${C.teal}33`, fontFamily: mono, fontSize: vertical ? 30 : 28, color: C.ink, lineHeight: 1.55 }}>
        <div style={{ fontFamily: body, fontWeight: 600, fontSize: 26, color: C.teal, marginBottom: 10 }}>recipe · rcp_3f1c…</div>
        <div><span style={{ color: C.muted }}>model </span>{hire.tag}</div>
        <div><span style={{ color: C.muted }}>task  </span>{suite}</div>
        <div><span style={{ color: C.muted }}>score </span><span style={{ color: C.teal, fontWeight: 600 }}>{Math.round(hire.score * 100)}%</span> · p50 {Math.round(hire.p50)} ms</div>
        <div><span style={{ color: C.muted }}>on    </span>{hardware}</div>
        <div style={{ fontFamily: body, fontSize: 24, color: C.muted, marginTop: 10 }}>no prompts · no outputs · no weights</div>
      </div>
      {NODES.map((n, i) => (
        <div key={n} style={{ position: "absolute", left: pos[i].x - 200, top: pos[i].y - 44, width: 400, height: 88, opacity: nodes[i],
          transform: `scale(${0.5 + 0.5 * nodes[i]})`, display: "flex", alignItems: "center", gap: 14, background: "#0A1020",
          border: `2px solid ${C.line}`, borderRadius: 18, padding: "0 22px", fontFamily: body, fontSize: 26, color: C.ink }}>
          <span style={{ color: C.teal, fontFamily: display, fontWeight: 800, fontSize: 34 }}>✓</span>{n}
        </div>
      ))}
      <div style={{ position: "absolute", left: vertical ? 90 : 120, top: vertical ? 1700 : height - 170, fontFamily: display, fontWeight: 600,
        fontSize: vertical ? 50 : 52, color: C.ink, opacity: interpolate(frame, [100, 120], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) }}>
        Others reproduce it. Every agent gets smarter.
      </div>
    </AbsoluteFill>
  );
};

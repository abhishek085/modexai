import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { LogoMark, useLayout } from "../components/Layout";
import { useSpringIn } from "../hooks/useSpringIn";
import { C, display } from "../theme";

/** 7–10.5s · solution reveal: a blue iris opens on the mark, then the promise. */
export const Reveal: React.FC = () => {
  const frame = useCurrentFrame();
  const { vertical, width, height } = useLayout();
  const iris = useSpringIn(0, 24);
  const mark = useSpringIn(8);
  const word = useSpringIn(16);
  const r = Math.hypot(width, height) * iris;
  const settle = interpolate(frame, [30, 60], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      <div style={{ position: "absolute", left: width / 2 - r, top: height / 2 - r, width: r * 2, height: r * 2, borderRadius: "50%",
        background: C.blue, opacity: 0.18 + 0.82 * settle }} />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 36, zIndex: 1 }}>
        <div style={{ transform: `scale(${0.4 + 0.6 * mark}) rotate(${(1 - mark) * -20}deg)`, opacity: mark }}>
          <LogoMark size={vertical ? 230 : 200} />
        </div>
        <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 150 : 140, letterSpacing: "-0.05em", color: C.white,
          opacity: word, transform: `translateY(${(1 - word) * 40}px)` }}>
          modex<span style={{ color: C.teal }}>ai</span>
        </div>
        <div style={{ marginTop: 10, width: vertical ? 900 : 1300 }}>
          <Kinetic text="Your agent hires a local apprentice." size={vertical ? 72 : 76} align="center" start={0.9} highlight={["local"]} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

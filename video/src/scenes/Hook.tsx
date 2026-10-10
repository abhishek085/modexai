import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Kinetic } from "../components/Kinetic";
import { useLayout } from "../components/Layout";
import { useSpringIn } from "../hooks/useSpringIn";
import { body, C, display, mono } from "../theme";

const TICKETS = [
  ["Sarah Connor", "sarah.connor@skynet.io", "refund my duplicate charge"],
  ["Marco Rossi", "+39 055 123 4567", "app crashes on login"],
  ["Priya N.", "priya@lumenlabs.io", "cancel before renewal"],
  ["Dev Patel", "dev.p@acme.com", "add SSO please"],
  ["Emma Novak", "IBAN …0532 0130", "wrong VAT on invoice"],
  ["Jordan Lee", "jordan@fastmail.com", "reset my password"],
];

/** 0–3s · pattern interrupt: private tickets streaming into a cloud API while a meter spins. */
export const Hook: React.FC = () => {
  const frame = useCurrentFrame();
  const { vertical, width, height } = useLayout();
  const tokens = Math.floor(interpolate(frame, [0, 90], [182_340, 1_284_211]));
  const cloudIn = useSpringIn(0);
  const counterIn = useSpringIn(10);
  const laneX = vertical ? width / 2 : width * 0.66;
  return (
    <AbsoluteFill>
      {/* the cloud everything disappears into */}
      <div style={{ position: "absolute", left: laneX - 150, top: vertical ? 590 : 70, width: 300, textAlign: "center",
        transform: `scale(${0.6 + 0.4 * cloudIn})`, opacity: cloudIn }}>
        <svg width="300" height="170" viewBox="0 0 300 170">
          <path d="M80 150h150a55 55 0 0 0 0-110 75 75 0 0 0-140-10A60 60 0 0 0 80 150z" fill="none" stroke={C.red} strokeWidth="8" />
        </svg>
        <div style={{ fontFamily: mono, fontSize: 28, color: C.red, marginTop: -8 }}>frontier API</div>
      </div>
      {/* tickets flying up */}
      {TICKETS.map((t, i) => {
        // Tickets rise through their own lane and vanish into the cloud — never across the text.
        const laneTop = vertical ? 800 : 290;
        const laneBottom = vertical ? 1400 : 1000;
        const range = laneBottom - laneTop + 160;
        const cycle = (frame * 13 + i * (range / TICKETS.length)) % range;
        const y = laneBottom - cycle;
        const op = interpolate(cycle, [0, 70, range - 200, range - 90], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        const scale = interpolate(cycle, [range - 330, range - 90], [1, 0.4], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        const x = laneX - 330 + Math.sin(i * 1.7) * (vertical ? 70 : 140);
        return (
          <div key={i} style={{ position: "absolute", left: x, top: y, width: 660, opacity: op, transform: `scale(${scale})`,
            background: C.night2, border: `2px solid ${C.line}`, borderRadius: 22, padding: "20px 26px", fontFamily: body }}>
            <div style={{ fontSize: 30, color: C.ink, fontWeight: 600 }}>
              <span style={{ background: `${C.red}33`, color: "#FFB3B0", padding: "0 6px", borderRadius: 6 }}>{t[0]}</span>{"  "}
              <span style={{ fontFamily: mono, fontSize: 24, color: "#FFB3B0" }}>{t[1]}</span>
            </div>
            <div style={{ fontSize: 28, color: C.muted, marginTop: 6 }}>“{t[2]}”</div>
          </div>
        );
      })}
      {/* headline */}
      <div style={{ position: "absolute", left: vertical ? 90 : 120, top: vertical ? 230 : 300, width: vertical ? 900 : 760 }}>
        <Kinetic text="Your agent sends every tiny task to the cloud." size={vertical ? 92 : 96} danger={["cloud"]} />
      </div>
      {/* meter */}
      <div style={{ position: "absolute", left: vertical ? 90 : 120, top: vertical ? 1530 : 760, opacity: counterIn,
        transform: `translateY(${(1 - counterIn) * 30}px)` }}>
        <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 120 : 110, color: C.red, letterSpacing: "-0.04em",
          fontVariantNumeric: "tabular-nums" }}>{tokens.toLocaleString("en-US")}</div>
        <div style={{ fontFamily: body, fontSize: 34, color: C.muted }}>tokens billed — to sort support tickets</div>
      </div>
    </AbsoluteFill>
  );
};

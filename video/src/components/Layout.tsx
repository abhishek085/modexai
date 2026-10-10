import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C } from "../theme";

export const useLayout = () => {
  const { width, height } = useVideoConfig();
  const vertical = height > width;
  return { vertical, width, height };
};

/** Persistent backdrop: blue-black field, faint measurement grid, two slow glows. */
export const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const drift = Math.sin(frame / 90) * 60;
  return (
    <AbsoluteFill style={{ background: C.night, overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${C.line}55 1px, transparent 1px), linear-gradient(90deg, ${C.line}55 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
          maskImage: "radial-gradient(ellipse at 50% 45%, black 30%, transparent 75%)",
          WebkitMaskImage: "radial-gradient(ellipse at 50% 45%, black 30%, transparent 75%)",
          opacity: 0.6,
        }}
      />
      <div style={{ position: "absolute", width: width * 0.9, height: width * 0.9, left: -width * 0.25 + drift, top: height * 0.08,
        background: `radial-gradient(circle, ${C.blue}40, transparent 65%)`, filter: "blur(20px)" }} />
      <div style={{ position: "absolute", width: width * 0.8, height: width * 0.8, right: -width * 0.3 - drift, bottom: height * 0.05,
        background: `radial-gradient(circle, ${C.teal}30, transparent 65%)`, filter: "blur(20px)" }} />
    </AbsoluteFill>
  );
};

/** Fades a whole scene in and out at its edges so cuts feel intentional, not abrupt. */
export const SceneFade: React.FC<{ duration: number; children: React.ReactNode; fadeIn?: number; fadeOut?: number }> = ({
  duration, children, fadeIn = 6, fadeOut = 8,
}) => {
  const frame = useCurrentFrame();
  const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
  const inO = fadeIn > 0 ? interpolate(frame, [0, fadeIn], [0, 1], clamp) : 1;
  const outO = interpolate(frame, [duration - fadeOut, duration], [1, 0], clamp);
  const opacity = Math.min(inO, outO);
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};

export const LogoMark: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 64 64">
    <rect width="64" height="64" rx="14" fill={C.blue} />
    <path d="M16 44V20l16 14 16-14v24" fill="none" stroke="#fff" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="48" cy="44" r="5" fill={C.teal} />
  </svg>
);

import React from "react";
import { C, mono } from "../theme";

export const Terminal: React.FC<{ title: string; width: number; children: React.ReactNode; style?: React.CSSProperties }> = ({ title, width, children, style }) => (
  <div style={{ width, background: "#0A1020", border: `2px solid ${C.line}`, borderRadius: 28, overflow: "hidden",
    boxShadow: `0 40px 120px ${C.blue}33, 0 0 0 1px #00000040`, ...style }}>
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "20px 26px", borderBottom: `2px solid ${C.line}`, background: "#0E1629" }}>
      {["#FF5F57", "#FEBC2E", "#28C840"].map((c) => <div key={c} style={{ width: 18, height: 18, borderRadius: 9, background: c }} />)}
      <div style={{ marginLeft: 16, fontFamily: mono, fontSize: 26, color: C.muted }}>{title}</div>
    </div>
    <div style={{ padding: "28px 30px" }}>{children}</div>
  </div>
);

import React from "react";
import { AbsoluteFill } from "remotion";
import { LogoMark, useLayout } from "../components/Layout";
import { useReveal, useSpringIn } from "../hooks/useSpringIn";
import { body, C, display, mono } from "../theme";

/** 27.5–30s · call to action. */
export const CTA: React.FC = () => {
  const { vertical } = useLayout();
  const mark = useSpringIn(0);
  const l1 = useReveal(6);
  const l2 = useReveal(14);
  const l3 = useReveal(22);
  const l4 = useReveal(30);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center", gap: 40, padding: 80 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 30, transform: `scale(${0.7 + 0.3 * mark})`, opacity: mark }}>
        <LogoMark size={vertical ? 150 : 130} />
        <div style={{ fontFamily: display, fontWeight: 800, fontSize: vertical ? 130 : 120, letterSpacing: "-0.05em", color: C.white }}>
          modex<span style={{ color: C.teal }}>ai</span>
        </div>
      </div>
      <div style={{ ...l1, fontFamily: display, fontWeight: 600, fontSize: vertical ? 70 : 66, color: C.ink, letterSpacing: "-0.03em" }}>
        What runs locally, measured.
      </div>
      <div style={{ ...l2, fontFamily: mono, fontSize: vertical ? 30 : 32, color: C.ink, background: "#0A1020", border: `2px solid ${C.line}`,
        borderRadius: 18, padding: "22px 30px", maxWidth: vertical ? 940 : 1500 }}>
        <span style={{ color: C.teal }}>claude mcp add</span> modexai -- uvx --from git+https://github.com/abhishek085/modexai modexai mcp
      </div>
      <div style={{ ...l3, fontFamily: body, fontWeight: 600, fontSize: vertical ? 46 : 44, color: C.white, background: C.blue,
        borderRadius: 999, padding: "20px 44px" }}>
        ★ Star on GitHub · abhishek085/modexai
      </div>
      <div style={{ ...l4, fontFamily: body, fontSize: vertical ? 32 : 30, color: C.muted }}>open source · local-first · a Nokast tool</div>
    </AbsoluteFill>
  );
};

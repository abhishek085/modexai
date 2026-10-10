import gsap from "gsap";
import { useMemo } from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";

type Opts = { start?: number; each?: number; duration?: number; ease?: string };

/**
 * Deterministic GSAP stagger for Remotion.
 *
 * Remotion renders frames out of order and in parallel, so real-time animation libraries can't drive it.
 * GSAP can: we build a *paused* timeline that tweens plain objects, then seek it to `frame / fps` on
 * every render. The same frame always yields the same values.
 *
 * Returns one 0→1 progress value per item.
 */
export const useGsapStagger = (count: number, { start = 0, each = 0.12, duration = 0.9, ease = "power3.out" }: Opts = {}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const tl = useMemo(() => {
    const targets = Array.from({ length: count }, () => ({ p: 0 }));
    const timeline = gsap.timeline({ paused: true });
    timeline.to(targets, { p: 1, duration, ease, stagger: each }, 0);
    return { timeline, targets };
  }, [count, each, duration, ease]);
  tl.timeline.seek(Math.max(0, frame / fps - start), true);
  return tl.targets.map((t) => t.p);
};

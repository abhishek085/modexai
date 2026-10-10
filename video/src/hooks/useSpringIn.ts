import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { SPRING } from "../theme";

/** 0 → 1 spring that starts `delay` frames into the current Sequence. */
export const useSpringIn = (delay = 0, durationInFrames?: number): number => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - delay, fps, config: SPRING, durationInFrames });
};

/** Spring that eases a card/line in: returns opacity + translate + scale styles. */
export const useReveal = (delay = 0, distance = 40) => {
  const p = useSpringIn(delay);
  return {
    opacity: p,
    transform: `translateY(${(1 - p) * distance}px) scale(${0.96 + 0.04 * p})`,
  } as const;
};

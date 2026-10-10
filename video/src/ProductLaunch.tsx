import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile } from "remotion";
import { Backdrop, SceneFade } from "./components/Layout";
import { Agitate } from "./scenes/Agitate";
import { CTA } from "./scenes/CTA";
import { Delegate } from "./scenes/Delegate";
import { Hook } from "./scenes/Hook";
import { Race } from "./scenes/Race";
import { Reveal } from "./scenes/Reveal";
import { Share } from "./scenes/Share";
import type { LaunchProps } from "./types";

/** Scene timing in frames @30fps (900 frames = 30 s). PAS: problem → agitate → solution → proof → CTA. */
export const SCENES = [
  { id: "hook", from: 0, dur: 90 },
  { id: "agitate", from: 90, dur: 120 },
  { id: "reveal", from: 210, dur: 105 },
  { id: "race", from: 315, dur: 240 },
  { id: "delegate", from: 555, dur: 120 },
  { id: "share", from: 675, dur: 150 },
  { id: "cta", from: 825, dur: 75 },
] as const;

const SFX: { at: number; src: string; volume?: number }[] = [
  { at: 0, src: "sfx/thump.wav", volume: 0.9 },
  { at: 90, src: "sfx/whoosh.wav", volume: 0.6 },
  { at: 210, src: "sfx/rise.wav", volume: 0.7 },
  { at: 222, src: "sfx/chime.wav", volume: 0.7 },
  { at: 315, src: "sfx/whoosh.wav", volume: 0.5 },
  { at: 465, src: "sfx/chime.wav", volume: 0.8 },
  { at: 555, src: "sfx/whoosh.wav", volume: 0.5 },
  { at: 675, src: "sfx/whoosh.wav", volume: 0.5 },
  { at: 735, src: "sfx/ticks.wav", volume: 0.5 },
  { at: 825, src: "sfx/thump.wav", volume: 0.7 },
  { at: 835, src: "sfx/chime.wav", volume: 0.8 },
];

export const ProductLaunch: React.FC<LaunchProps> = (props) => {
  const seq = (id: (typeof SCENES)[number]["id"], node: React.ReactNode) => {
    const sc = SCENES.find((x) => x.id === id)!;
    return (
      <Sequence from={sc.from} durationInFrames={sc.dur} name={id}>
        <SceneFade duration={sc.dur} fadeIn={id === "hook" ? 0 : 6}>{node}</SceneFade>
      </Sequence>
    );
  };
  return (
    <AbsoluteFill>
      <Backdrop />
      {seq("hook", <Hook />)}
      {seq("agitate", <Agitate />)}
      {seq("reveal", <Reveal />)}
      {seq("race", <Race {...props} />)}
      {seq("delegate", <Delegate {...props} />)}
      {seq("share", <Share {...props} />)}
      {seq("cta", <CTA />)}
      <Audio src={staticFile("sfx/bed.wav")} volume={(f) => Math.min(1, f / 20) * 0.35} />
      {SFX.map((x, i) => (
        <Sequence key={i} from={x.at} durationInFrames={60} layout="none">
          <Audio src={staticFile(x.src)} volume={x.volume ?? 0.7} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

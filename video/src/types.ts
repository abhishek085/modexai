export type RaceRow = { tag: string; score: number; p50: number; size: number };

export type LaunchProps = {
  /** Real measurements from the ModexAI recipe index (see scripts/video_data.py). */
  race: RaceRow[];
  hire: RaceRow;
  suite: string;
  hardware: string;
  /** Hiring bar (0–1): the hire is the smallest model at or above it. */
  bar: number;
};

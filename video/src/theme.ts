import { loadFont as loadDisplay } from "@remotion/google-fonts/BricolageGrotesque";
import { loadFont as loadBody } from "@remotion/google-fonts/IBMPlexSans";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

export const display = loadDisplay("normal", { weights: ["600", "800"], subsets: ["latin"] }).fontFamily;
export const body = loadBody("normal", { weights: ["400", "500", "600"], subsets: ["latin"] }).fontFamily;
export const mono = loadMono("normal", { weights: ["400", "600"], subsets: ["latin"] }).fontFamily;

// Nokast palette, tuned for a dark launch film.
export const C = {
  night: "#0D1424",
  night2: "#131C30",
  line: "#24304A",
  ink: "#E7ECF5",
  muted: "#98A3B8",
  blue: "#2A6DFF",
  blueSoft: "#5B8DFF",
  teal: "#00C4B4",
  amber: "#F2B34B",
  red: "#FF5A55",
  white: "#FFFFFF",
};

// One spring config for every reveal, per the brief.
export const SPRING = { damping: 20, mass: 0.5 } as const;

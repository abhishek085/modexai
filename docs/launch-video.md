# ModexAI launch film: strategy, script and build

A 30-second launch explainer, made in two cuts from one Remotion project:

| Cut | Size | File |
|---|---|---|
| Vertical (TikTok / Reels / Shorts) | 1080×1920, 30 fps | [`assets/video/modexai-launch-9x16.mp4`](../assets/video/modexai-launch-9x16.mp4) |
| Wide (X, LinkedIn, YouTube, README) | 1920×1080, 30 fps | [`assets/video/modexai-launch-16x9.mp4`](../assets/video/modexai-launch-16x9.mp4) |

Source: [`video/`](../video). Re-render with `cd video && npm i && npm run render && npm run render:wide`.

Every number on screen in the "proof" scenes comes from a real recipe in the index (ticket-triage
suite, Apple M4 Pro · 24 GB), pulled by [`scripts/video_data.py`](../scripts/video_data.py). The film is
captions-first because most feeds autoplay muted. The voiceover below is written for a human or
text-to-speech recording laid over the same timing.

## Product context

- **Product:** ModexAI, an MCP server + CLI + open exchange.
- **Audience:** engineers building with agents (Claude Code, Cursor, Codex), local-AI tinkerers, privacy-minded teams.
- **Problem:** agents send every small, repetitive task (sorting tickets, pulling fields, redacting) to a
  frontier API. Each call is billed, and each one carries the user's data off the machine.
- **Breakthrough:** the agent hires its own local apprentice. It tests small models against its own
  answers, on the user's machine, and hands over the tasks they've proven they can do. Then it shares
  the finding as a recipe (no weights, no data) that other agents can reproduce.
- **CTA:** Star on GitHub, and `claude mcp add modexai …`
- **Aesthetic:** Nokast dark mode. Blue-black field, electric blue `#2A6DFF`, teal `#00C4B4` for
  "kept local / passes", red only for money and data leaving. High-contrast kinetic type (Bricolage
  Grotesque), real terminal UI (JetBrains Mono).

<strategy>
**Framework: PAS with a visual reveal.** Problem, Agitate, Solution, then proof, network effect, CTA.

**Hook (0–3 s), the pattern interrupt.** No logo and no intro. Frame 1 is already moving: real-looking
support tickets, names and emails highlighted in red, stream upward into a red "frontier API" cloud while
a token meter spins past a million. The headline lands word by word: *"Your agent sends every tiny task to
the cloud."* The tension is immediate and visual. Money is burning, and private data is visibly leaving.

**Agitate (3–7 s).** Three costs stack in with spring physics: billed by the token, on someone else's
server, stops when the API does. Then the turn: *"Does sorting tickets really need a frontier model?"*

**Solution reveal (7–10.5 s).** A blue iris opens on the mark: *"Your agent hires a local apprentice."*

**Show, don't tell (10.5–18.5 s).** The real CLI race board (`modexai hire ticket-triage`) with real
measured scores filling against a dashed 80% bar. The verdict card springs in: *Hire gemma3:1b*, the
smallest model that clears it — while a model three times its size scores only five points higher. The UI state transition is the argument.

**Payoff (18.5–22.5 s).** A counter runs to 300 tickets handled locally. Three chips pop: *0 tokens
billed · 0 bytes sent · works offline.* *"The work stays home."*

**Network effect (22.5–27.5 s).** The finding leaves as a recipe card (model, task, score, hardware; *no
prompts, no outputs, no weights*). Lines draw out to other machines that tick ✓. *"Others reproduce it.
Every agent gets smarter."* This beat explains why to star the repo rather than just use the tool.

**CTA (27.5–30 s).** Wordmark, *"What runs locally, measured."*, the install command, and *★ Star on GitHub*.

**Safe zones.** In the vertical cut, the top 220 px and bottom 300 px hold no essential text (platform
UI). Headlines sit at 230–420 px and the CTA is centred. The wide cut re-flows the same scenes into
two-column layouts using the `useLayout()` hook, with no separate assets.

**Retention devices.** A new visual state every 1–1.5 s; numbers that move (meter, scores, counter); one
colour rule the eye learns fast (red = leaving / billed, teal = local / passes); a reveal at 7 s, before
the typical swipe-away point.
</strategy>

<script_breakdown>
  <scene number="1" duration="0-3s">
    <visual_description>Blue-black field with a faint measurement grid. Six support tickets (name + email highlighted red, a one-line request in grey) drift upward and shrink into a red outlined cloud labelled "frontier API". Bottom-left, a red token meter counts up fast from 182,340 to 1,284,211.</visual_description>
    <motion_graphic>Headline words rise in on a GSAP back.out stagger (0.07 s each). The cloud scales in on spring(damping 20, mass 0.5). Tickets loop on a continuous upward path with opacity and scale falloff near the cloud. Meter digits use tabular figures so they don't jitter.</motion_graphic>
    <voiceover>"Your agent sends every tiny task to the cloud."</voiceover>
    <sfx>Sub-bass thump on frame 0 (55 Hz, 0.5 s decay). Low ambient pad fades in under everything.</sfx>
  </scene>
  <scene number="2" duration="3-7s">
    <visual_description>Three statements stack on the left, each with a red rule: "Every ticket → billed by the token", "Every customer email → on someone else's server", "Every outage → your agent stops working". The question appears on the right (wide) or below (vertical).</visual_description>
    <motion_graphic>Each line springs up 50 px with 16-frame offsets. The question types in word by word at 2.0 s, with "really" in teal.</motion_graphic>
    <voiceover>"Every call is billed. Every email lives on someone else's server. Does sorting tickets really need a frontier model?"</voiceover>
    <sfx>Soft pink-noise whoosh on the cut.</sfx>
  </scene>
  <scene number="3" duration="7-10.5s">
    <visual_description>A blue circle irises open from centre and settles to a soft tint. The ModexAI mark rotates and scales into place, the wordmark rises and the tagline lands: "Your agent hires a local apprentice."</visual_description>
    <motion_graphic>Iris radius is driven by a 24-frame spring. The mark goes from scale 0.4 to 1 with -20° to 0° rotation. Wordmark at +16 frames. Tagline GSAP stagger at 0.9 s, with "local" in teal.</motion_graphic>
    <voiceover>"Meet ModexAI. Your agent hires a local apprentice."</voiceover>
    <sfx>Filtered noise riser into a two-note chime (A5 + E6).</sfx>
  </scene>
  <scene number="4" duration="10.5-18.5s">
    <visual_description>A terminal window. `$ modexai hire ticket-triage` types out. Four model rows race: progress bars fill against a dashed 80% bar (the default hiring bar) while scores count up to real measured values. The smallest passing model's row glows teal. A teal verdict card springs in: "Hire gemma3:1b — 80% · 156 ms · 0.81 GB — the smallest model that clears the bar". The race rows are qwen3:0.6b 20%, granite4:350m 50%, gemma3:1b 80%, qwen3:4b-instruct 85% — real ticket-triage results. Footer: "running on Apple M4 Pro · 24 GB · nothing leaves it".</visual_description>
    <motion_graphic>Typing driven by frame interpolation (frames 12–40). Bars use a GSAP power2.inOut stagger (0.25 s apart, 2.6 s each) seeked to frame/fps. Bars are blue while running, then teal (pass), amber (close) or grey. The verdict springs in at frame 150 from +60 px and scale 0.9.</motion_graphic>
    <voiceover>"It tests small models against your agent's own answers, right on your machine, and hires the smallest one that's good enough."</voiceover>
    <sfx>Whoosh on entry; chime when the verdict lands.</sfx>
  </scene>
  <scene number="5" duration="18.5-22.5s">
    <visual_description>A giant teal counter runs 0 → 300, "new tickets handled by gemma3:1b on Apple M4 Pro · 24 GB". Three outlined chips pop in: "0 tokens billed", "0 bytes sent", "works offline". Headline: "The work stays home."</visual_description>
    <motion_graphic>Counter interpolates over frames 8–80. Chips use a GSAP back.out(2) stagger. Headline stagger at 1.9 s.</motion_graphic>
    <voiceover>"Then the work stays home. No tokens billed, no data sent."</voiceover>
    <sfx>Whoosh on entry.</sfx>
  </scene>
  <scene number="6" duration="22.5-27.5s">
    <visual_description>A recipe card slides in (model, task, score, hardware; "no prompts · no outputs · no weights"). Teal lines draw out to six machine nodes (RTX 4090, MacBook Air M2, Raspberry Pi 5, Framework, Mac mini M4, Jetson Orin), each ticking ✓. Closing line: "Others reproduce it. Every agent gets smarter."</visual_description>
    <motion_graphic>The card springs in with a slight -4° to 0° rotation. Lines animate stroke-dashoffset over frames 50–100. Nodes use a GSAP back.out(1.8) stagger at 0.16 s. The machines are illustrative; the card's numbers are real.</motion_graphic>
    <voiceover>"And it shares a recipe — not weights, not data — so the next agent starts ahead."</voiceover>
    <sfx>Whoosh, then a run of soft ticks as nodes confirm.</sfx>
  </scene>
  <scene number="7" duration="27.5-30s">
    <visual_description>Mark + wordmark, "What runs locally, measured.", the install command in a terminal pill, a blue "★ Star on GitHub · abhishek085/modexai" button, and "open source · local-first · a Nokast tool".</visual_description>
    <motion_graphic>Four springs at 8-frame offsets (opacity, +40 px, scale 0.96 → 1).</motion_graphic>
    <voiceover>"ModexAI. What runs locally, measured. Star it on GitHub."</voiceover>
    <sfx>Thump plus chime; the pad fades out over the last 2.5 s.</sfx>
  </scene>
</script_breakdown>

<ai_video_prompts>
Use these to generate optional B-roll or alternate hero shots in Runway Gen-3 / Sora / Kling. Generate
at 1080×1920 for the vertical cut (or 1920×1080) at 24–30 fps, 5 s clips, then composite under the
Remotion UI layers. Shared negative prompt: *no text, no logos, no watermark, no people's faces, no
lens flares, no warm/orange grading.*

**Shot 1 — Hook (0–3 s)**
[Dozens of translucent glass ticket cards streaming upward from a dark desk into a glowing red cloud of light, data particles trailing], [Camera: low-angle tilt-up, slow push-in 15% over 3 s, 35 mm], [Lighting: dark mode studio lighting, deep blue-black ambience (#0D1424), single hard red rim light from above (#FF5A55), subtle volumetric haze], [Style: 8k resolution, photorealistic, 30fps, octane render, shallow depth of field, motion blur on cards]

**Shot 2 — Agitate (3–7 s)**
[A cascade of tiny glowing price tags and padlock icons drifting away from a laptop toward a distant server farm on the horizon], [Camera: slow 90° lateral orbit around the laptop, locked focus on the screen edge], [Lighting: cold blue key light, red accents on the drifting tags, fog layers in the background], [Style: 8k resolution, photorealistic, 30fps, octane render, anamorphic bokeh]

**Shot 3 — Reveal (7–10.5 s)**
[A perfectly round iris of electric blue light (#2A6DFF) opening in total darkness, revealing a clean glowing tile with a white "M" path and a small teal dot], [Camera: static, then a gentle 5% dolly-in as the iris completes], [Lighting: volumetric blue glow, soft bloom, teal (#00C4B4) specular kiss on the tile edge], [Style: 8k resolution, 30fps, octane render, minimal, symmetrical composition]

**Shot 4 — Race (10.5–18.5 s)**
[Macro close-up of an Apple Silicon laptop keyboard and screen glow; reflected on the aluminium, four horizontal light bars race left to right and one turns teal], [Camera: FPV-style slow slide along the keyboard toward the screen, rack focus from keys to screen reflection], [Lighting: dark mode, screen as the only key light, teal highlight bloom when the winning bar completes], [Style: 8k resolution, photorealistic, 60fps (conform to 30), octane render, macro lens, crisp reflections]

**Shot 5 — Delegate (18.5–22.5 s)**
[A small laptop on a wooden desk at night, a soft teal aura pulsing from the chip area while a stack of paper tickets on the desk dissolves into light that flows *into* the laptop, not out], [Camera: slow 180° orbit around the laptop at desk height], [Lighting: warm-neutral practical lamp at 15%, dominant cool blue ambience, teal under-glow], [Style: 8k resolution, photorealistic, 30fps, octane render, cosy but clean]

**Shot 6 — Network (22.5–27.5 s)**
[A single glowing teal card in the centre of a dark space; thin light threads shoot out to six small hovering devices (a desktop GPU tower, a thin laptop, a Raspberry Pi board, a mini PC, an edge module, a Mac mini), each lighting up with a soft teal check-glow], [Camera: pull-back crane move from the card to a wide shot revealing all six devices], [Lighting: volumetric neon glow along the threads, deep blue-black void, gentle particles], [Style: 8k resolution, 30fps, octane render, isometric-leaning composition, clean product-viz look]

**Shot 7 — CTA (27.5–30 s)**
[Abstract slow-moving gradient of electric blue and teal light ribbons on deep blue-black, leaving clear negative space in the centre for typography], [Camera: static, ribbons drift slowly left to right], [Lighting: soft bloom, no hard highlights], [Style: 8k resolution, 30fps, octane render, minimal background plate]
</ai_video_prompts>

## Remotion build

The brief asked for a Remotion composition named `ProductLaunch` at 1080×1920, 30 fps, spring dynamics
`spring({ damping: 20, mass: 0.5 })` on every card and text entrance, and GSAP or Framer Motion for
staggered timing. That's what [`video/src`](../video/src) contains:

```
video/src
├── index.ts                 registerRoot
├── Root.tsx                 <Composition id="ProductLaunch" 1080×1920 @30 /> + "ProductLaunchWide" 1920×1080
├── ProductLaunch.tsx        scene timeline (SCENES), SFX cues, music bed
├── types.ts                 LaunchProps { race, hire, suite, hardware }
├── data.json                real measurements (generated by scripts/video_data.py)
├── theme.ts                 palette, fonts (@remotion/google-fonts), SPRING = { damping: 20, mass: 0.5 }
├── hooks/
│   ├── useSpringIn.ts       spring({ frame, fps, config: SPRING }) → 0..1, plus useReveal() styles
│   └── useGsapStagger.ts    paused GSAP timeline seeked to frame/fps → per-item progress
├── components/
│   ├── Layout.tsx           Backdrop, SceneFade, LogoMark, useLayout() (vertical vs wide)
│   ├── Kinetic.tsx          word-by-word kinetic headline (GSAP stagger)
│   └── Terminal.tsx         macOS-style terminal frame
└── scenes/                  Hook, Agitate, Reveal, Race, Delegate, Share, CTA
```

**Render props.** Both compositions take `LaunchProps`:

```ts
type RaceRow = { tag: string; score: number; p50: number; size: number };
type LaunchProps = { race: RaceRow[]; hire: RaceRow; suite: string; hardware: string };
```

Override from the CLI to make a cut for another task or machine:

```bash
npx remotion render src/index.ts ProductLaunch out/custom.mp4 --props='{"race":[...],"hire":{...},"suite":"json-extraction","hardware":"RTX 4090 24 GB"}'
```

**Why GSAP and not Framer Motion.** Remotion renders frames out of order and in parallel, so any
library that animates against wall-clock time produces flicker or wrong frames. GSAP works if you never
let it play: `useGsapStagger` builds a paused timeline that tweens plain objects and calls
`timeline.seek(frame / fps)` on every render, so a given frame always yields the same values. Framer
Motion has no equivalent seek API, which is why it isn't used.

**Rendering on a machine without Chrome Headless Shell.** Pass any Chromium binary:
`--browser-executable="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"`.

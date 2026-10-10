import React from "react";
import { Composition } from "remotion";
import data from "./data.json";
import { ProductLaunch } from "./ProductLaunch";
import { SocialCard } from "./SocialCard";
import type { LaunchProps } from "./types";

const defaultProps = data as LaunchProps;

export const RemotionRoot: React.FC = () => (
  <>
    {/* 9:16 for TikTok / Reels / Shorts — key content kept inside the platform safe zone */}
    <Composition id="ProductLaunch" component={ProductLaunch} width={1080} height={1920} fps={30} durationInFrames={900}
      defaultProps={defaultProps} />
    {/* 16:9 for X, LinkedIn, YouTube and the README */}
    <Composition id="ProductLaunchWide" component={ProductLaunch} width={1920} height={1080} fps={30} durationInFrames={900}
      defaultProps={defaultProps} />
    {/* GitHub social preview (Settings → Social preview) */}
    <Composition id="SocialCard" component={SocialCard} width={1280} height={640} fps={30} durationInFrames={1}
      defaultProps={defaultProps} />
  </>
);

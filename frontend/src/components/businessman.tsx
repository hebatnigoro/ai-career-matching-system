"use client";

import { useEffect, useRef, useState } from "react";

export type Expression = "happy" | "thinking" | "focused";

type Props = {
  mouseX: number;
  mouseY: number;
  expression?: Expression;
  scale?: number;
};

const C = {
  border: "#1e1a3a",
  skin: "#F87171",
  skinDark: "#DC2626",
  hair: "#1F2433",
  shirt: "#FFFFFF",
  jacket: "#0EA5E9",
  jacketDark: "#0284C7",
  pants: "#1F2433",
  shoes: "#0F172A",
  briefcase: "#FBBF24",
  briefcaseDark: "#B45309",
  white: "#FFFFFF",
};

export function Businessman({ mouseX, mouseY, expression = "happy", scale = 1 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ ex: 0, ey: 0, lean: 0, headShift: 0, breathe: 0, t: 0 });
  const [s, setS] = useState({ ex: 0, ey: 0, lean: 0, headShift: 0, breathe: 0, t: 0 });
  const [blink, setBlink] = useState(false);
  const startRef = useRef<number>(0);

  useEffect(() => {
    let raf = 0;
    if (startRef.current === 0) startRef.current = performance.now();
    function tick() {
      const el = ref.current;
      if (el && typeof window !== "undefined") {
        const r = el.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const dx = (mouseX - cx) / (window.innerWidth * 0.5);
        const dy = (mouseY - cy) / (window.innerHeight * 0.5);
        const t = (performance.now() - startRef.current) / 1000;
        const breathTarget = Math.sin(t * 1.6) * 1.2;
        const T = { ex: dx * 3, ey: dy * 2.5, lean: dx * 5, headShift: dx * 7 };
        const SP = { ex: 0.12, ey: 0.11, lean: 0.055, headShift: 0.065 };
        const cur = stateRef.current;
        const next = {
          ex: cur.ex + (T.ex - cur.ex) * SP.ex,
          ey: cur.ey + (T.ey - cur.ey) * SP.ey,
          lean: cur.lean + (T.lean - cur.lean) * SP.lean,
          headShift: cur.headShift + (T.headShift - cur.headShift) * SP.headShift,
          breathe: breathTarget,
          t,
        };
        stateRef.current = next;
        setS(next);
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mouseX, mouseY]);

  useEffect(() => {
    let cancelled = false;
    function loop() {
      if (cancelled) return;
      setBlink(true);
      setTimeout(() => !cancelled && setBlink(false), 130);
      const next = 2800 + Math.random() * 2400;
      setTimeout(loop, next);
    }
    const start = setTimeout(loop, 1500);
    return () => {
      cancelled = true;
      clearTimeout(start);
    };
  }, []);

  const cl = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
  const ex = cl(s.ex, -3, 3);
  const ey = cl(s.ey, -2.5, 2.5);
  const lean = cl(s.lean, -5, 5);
  const hS = cl(s.headShift, -7, 7);
  const breathe = s.breathe;

  const wave = expression === "happy" ? Math.sin(s.t * 3.2) * 12 : 0;
  const thinking = expression === "thinking";
  const focused = expression === "focused";

  // Eye centers — simple dot eyes, larger spacing for bigger head
  const lEyeCx = 84;
  const rEyeCx = 116;
  const eyeCy = 92;
  const eyeR = 3.5;

  // Brow shape per expression — single curve per brow
  const browL =
    thinking
      ? "M 72 76 Q 84 70 92 74"
      : focused
      ? "M 72 80 L 92 76"
      : "M 72 78 Q 84 74 92 78";
  const browR =
    thinking
      ? "M 108 74 Q 116 78 128 78"
      : focused
      ? "M 108 76 L 128 80"
      : "M 108 78 Q 116 74 128 78";

  // Mouth per expression — simple flat-style curve
  let mouth: React.ReactElement;
  if (expression === "happy") {
    mouth = (
      <path
        d="M 90 112 Q 100 122 110 112"
        fill="none"
        stroke={C.border}
        strokeWidth="2.8"
        strokeLinecap="round"
      />
    );
  } else if (expression === "thinking") {
    mouth = (
      <path
        d="M 92 114 Q 100 112 108 116"
        fill="none"
        stroke={C.border}
        strokeWidth="2.8"
        strokeLinecap="round"
      />
    );
  } else {
    mouth = (
      <path
        d="M 92 115 Q 100 117 108 115"
        fill="none"
        stroke={C.border}
        strokeWidth="2.8"
        strokeLinecap="round"
      />
    );
  }

  const W = 200;
  const H = 320;

  return (
    <div
      ref={ref}
      style={{
        width: W * scale,
        height: H * scale,
        flexShrink: 0,
        userSelect: "none",
      }}
    >
      <svg
        width={W * scale}
        height={H * scale}
        viewBox={`0 0 ${W} ${H}`}
        overflow="visible"
        style={{ display: "block" }}
      >
        {/* Floor shadow */}
        <ellipse
          cx={100 + lean * 0.3}
          cy={310}
          rx={50}
          ry={5}
          fill={C.border}
          opacity="0.15"
        />

        <g transform={`translate(0, ${breathe})`}>
          <g transform={`rotate(${lean}, 100, 308)`}>
            {/* Legs */}
            <rect x={70} y={235} width={22} height={52} rx={11} fill={C.pants} />
            <rect x={108} y={235} width={22} height={52} rx={11} fill={C.pants} />
            {/* Shoes */}
            <ellipse cx={81} cy={290} rx={16} ry={6.5} fill={C.shoes} />
            <ellipse cx={119} cy={290} rx={16} ry={6.5} fill={C.shoes} />

            {/* Jacket — single flat shape, no lapels */}
            <path
              d="M 50 150
                 Q 50 132 68 130
                 L 132 130
                 Q 150 132 150 150
                 L 150 240
                 Q 150 248 142 248
                 L 58 248
                 Q 50 248 50 240 Z"
              fill={C.jacket}
            />

            {/* Shirt collar V — single white shape */}
            <path
              d="M 86 130 L 100 168 L 114 130 Z"
              fill={C.shirt}
            />

            {/* LEFT ARM — always at side */}
            <g>
              <rect x={48} y={150} width={24} height={86} rx={12} fill={C.jacket} />
              <circle cx={60} cy={244} r={11} fill={C.skin} />
            </g>

            {/* RIGHT ARM */}
            {expression === "focused" ? (
              <g>
                <rect x={128} y={150} width={24} height={70} rx={12} fill={C.jacket} />
                <circle cx={140} cy={224} r={11} fill={C.skin} />
                {/* Briefcase */}
                <rect x={148} y={218} width={42} height={30} rx={5} fill={C.briefcase} />
                <rect x={163} y={210} width={12} height={11} rx={3} fill="none" stroke={C.briefcaseDark} strokeWidth="2.5" />
                <rect x={148} y={232} width={42} height={3.5} fill={C.briefcaseDark} opacity="0.45" />
              </g>
            ) : expression === "happy" ? (
              <g transform={`rotate(${-30 + wave}, 140, 152)`}>
                <rect x={128} y={150} width={24} height={70} rx={12} fill={C.jacket} />
                <circle cx={140} cy={224} r={12} fill={C.skin} />
              </g>
            ) : (
              <g>
                <rect x={128} y={150} width={24} height={86} rx={12} fill={C.jacket} />
                <circle cx={140} cy={244} r={11} fill={C.skin} />
              </g>
            )}

            {/* Briefcase on ground when not focused */}
            {expression !== "focused" && (
              <g transform="translate(150, 252)">
                <rect x={0} y={0} width={36} height={26} rx={4} fill={C.briefcase} />
                <rect x={13} y={-7} width={10} height={9} rx={3} fill="none" stroke={C.briefcaseDark} strokeWidth="2.5" />
                <rect x={0} y={11} width={36} height={3} fill={C.briefcaseDark} opacity="0.45" />
              </g>
            )}

            {/* HEAD */}
            <g transform={`translate(${hS * 0.45}, 0)`}>
              {/* Neck */}
              <path
                d="M 90 124 L 90 138 Q 100 142 110 138 L 110 124 Z"
                fill={C.skin}
              />
              {/* Neck shadow */}
              <path
                d="M 90 124 Q 100 130 110 124 L 110 130 Q 100 134 90 130 Z"
                fill={C.skinDark}
                opacity="0.45"
              />

              {/* Ears */}
              <ellipse cx={50} cy={92} rx={6} ry={9} fill={C.skin} />
              <ellipse cx={150} cy={92} rx={6} ry={9} fill={C.skin} />

              {/* Head — large, rounder, flat-style */}
              <ellipse cx={100} cy={88} rx={48} ry={52} fill={C.skin} />

              {/* Hair — clean side-swept style, sits above the eyebrows */}
              <path
                d="M 54 70
                   L 54 46
                   Q 54 18 100 16
                   Q 146 18 146 46
                   L 146 70
                   Q 132 60 112 64
                   Q 100 72 86 62
                   Q 66 60 54 70 Z"
                fill={C.hair}
              />
              {/* Forelock — small swoop for character */}
              <path
                d="M 96 36 Q 116 30 130 50 Q 118 46 108 52 Q 100 50 96 36 Z"
                fill={C.hair}
              />

              {/* Cheek blush — subtle */}
              <ellipse cx={68} cy={108} rx={7} ry={4} fill={C.skinDark} opacity="0.45" />
              <ellipse cx={132} cy={108} rx={7} ry={4} fill={C.skinDark} opacity="0.45" />

              {/* Eyebrows */}
              <path d={browL} fill="none" stroke={C.border} strokeWidth="2.8" strokeLinecap="round" />
              <path d={browR} fill="none" stroke={C.border} strokeWidth="2.8" strokeLinecap="round" />

              {/* Eyes — simple dots, blink as a line */}
              {blink ? (
                <>
                  <line
                    x1={lEyeCx - 5}
                    y1={eyeCy}
                    x2={lEyeCx + 5}
                    y2={eyeCy}
                    stroke={C.border}
                    strokeWidth="2.8"
                    strokeLinecap="round"
                  />
                  <line
                    x1={rEyeCx - 5}
                    y1={eyeCy}
                    x2={rEyeCx + 5}
                    y2={eyeCy}
                    stroke={C.border}
                    strokeWidth="2.8"
                    strokeLinecap="round"
                  />
                </>
              ) : (
                <>
                  <circle
                    cx={lEyeCx + ex * 0.6}
                    cy={eyeCy + ey * 0.6}
                    r={focused ? 2.6 : eyeR}
                    fill={C.border}
                  />
                  <circle
                    cx={rEyeCx + ex * 0.6}
                    cy={eyeCy + ey * 0.6}
                    r={focused ? 2.6 : eyeR}
                    fill={C.border}
                  />
                </>
              )}

              {/* Glasses (focused only) — simple flat outline */}
              {focused && (
                <g>
                  <circle cx={lEyeCx} cy={eyeCy} r={11} fill="none" stroke={C.border} strokeWidth="2.5" />
                  <circle cx={rEyeCx} cy={eyeCy} r={11} fill="none" stroke={C.border} strokeWidth="2.5" />
                  <line x1={lEyeCx + 11} y1={eyeCy} x2={rEyeCx - 11} y2={eyeCy} stroke={C.border} strokeWidth="2.5" />
                </g>
              )}

              {/* Nose — simple curve */}
              <path
                d="M 98 100 Q 96 106 100 108"
                fill="none"
                stroke={C.border}
                strokeWidth="2"
                strokeLinecap="round"
              />

              {/* Mouth */}
              {mouth}

              {/* Thought bubble — thinking expression only */}
              {thinking && (
                <g>
                  <circle cx={158} cy={48} r={4} fill="#F1F5F9" stroke={C.border} strokeWidth="2" />
                  <circle cx={170} cy={36} r={6} fill="#F1F5F9" stroke={C.border} strokeWidth="2" />
                  <circle cx={186} cy={20} r={11} fill="#F1F5F9" stroke={C.border} strokeWidth="2.5" />
                  <text x={186} y={24} textAnchor="middle" fontSize="13" fontWeight="900" fill={C.border}>?</text>
                </g>
              )}
            </g>
          </g>
        </g>
      </svg>
    </div>
  );
}

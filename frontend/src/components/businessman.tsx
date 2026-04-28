"use client";

import { useEffect, useRef, useState } from "react";

type Props = { mouseX: number; mouseY: number };

export function Businessman({ mouseX, mouseY }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ ex: 0, ey: 0, lean: 0, headShift: 0, eyeY: 0 });
  const [s, setS] = useState(stateRef.current);

  useEffect(() => {
    let raf = 0;
    function tick() {
      const el = ref.current;
      if (el && typeof window !== "undefined") {
        const r = el.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const dx = (mouseX - cx) / (window.innerWidth * 0.5);
        const dy = (mouseY - cy) / (window.innerHeight * 0.5);
        const T = { ex: dx * 4, ey: dy * 3.5, lean: dx * 7, headShift: dx * 9, eyeY: dy * 2.5 };
        const SP = { ex: 0.12, ey: 0.11, lean: 0.055, headShift: 0.065, eyeY: 0.09 };
        const cur = stateRef.current;
        const next = {
          ex: cur.ex + (T.ex - cur.ex) * SP.ex,
          ey: cur.ey + (T.ey - cur.ey) * SP.ey,
          lean: cur.lean + (T.lean - cur.lean) * SP.lean,
          headShift: cur.headShift + (T.headShift - cur.headShift) * SP.headShift,
          eyeY: cur.eyeY + (T.eyeY - cur.eyeY) * SP.eyeY,
        };
        stateRef.current = next;
        setS({ ...next });
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mouseX, mouseY]);

  const cl = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
  const ex = cl(s.ex, -4, 4);
  const ey = cl(s.ey, -3.5, 3.5);
  const lean = cl(s.lean, -7, 7);
  const hS = cl(s.headShift, -9, 9);

  const PU = "#7c6fe0", WH = "#ffffff", YL = "#ffd166", BD = "#1e1a3a", PK = "#ff9bb5", MT = "#6dd5c0";

  const plx = cl(58 + ex, 58, 70), ply = cl(72 + ey, 72, 82);
  const prx = cl(98 + ex, 98, 110), pry = cl(72 + ey, 72, 82);

  return (
    <div ref={ref} style={{ width: 180, height: 300, flexShrink: 0 }}>
      <svg width="180" height="300" viewBox="0 0 180 300" overflow="visible">
        <rect x={44 + lean * 0.3} y="290" width="92" height="8" rx="4" fill="#c8c2f0" opacity="0.35" />
        <g transform={`rotate(${lean}, 90, 288)`}>
          <rect x="50" y="260" width="34" height="14" rx="4" fill={BD} />
          <rect x="96" y="260" width="34" height="14" rx="4" fill={BD} />
          <rect x="44" y="128" width="92" height="136" rx="6" fill={PU} stroke={BD} strokeWidth="2.5" />
          <rect x="76" y="128" width="28" height="108" rx="4" fill={WH} />
          <rect x="84" y="136" width="12" height="58" rx="6" fill={PK} stroke={BD} strokeWidth="1.5" />
          <rect x="82" y="132" width="16" height="14" rx="4" fill="#e07898" stroke={BD} strokeWidth="1.5" />
          <rect x="52" y="150" width="18" height="14" rx="3" fill={WH} stroke={BD} strokeWidth="1.5" />
          <rect x="54" y="153" width="8" height="7" rx="1" fill={MT} />
          <rect x="136" y="180" width="34" height="26" rx="4" fill={YL} stroke={BD} strokeWidth="2.5" />
          <rect x="148" y="173" width="10" height="9" rx="3" fill="none" stroke={BD} strokeWidth="2" />
          <rect x="136" y="192" width="34" height="3" fill={BD} />
          <rect x="152" y="188" width="6" height="9" rx="2" fill="#c8a020" />
          <g transform={`translate(${hS * 0.45}, 0)`}>
            <rect x="82" y="118" width="16" height="14" rx="3" fill={PU} stroke={BD} strokeWidth="2" />
            <rect x="44" y="38" width="92" height="84" rx="8" fill={YL} stroke={BD} strokeWidth="2.5" />
            <rect x="34" y="36" width="112" height="10" rx="4" fill={BD} />
            <rect x="54" y="0" width="72" height="40" rx="5" fill={BD} />
            <rect x="54" y="32" width="72" height="8" rx="3" fill={PK} />
            <rect x="56" y="68" width="28" height="24" rx="4" fill={WH} stroke={BD} strokeWidth="2" />
            <rect x={plx} y={ply} width="12" height="12" rx="3" fill={BD} />
            <rect x={plx + 2} y={ply + 2} width="4" height="4" rx="2" fill={WH} />
            <rect x="96" y="68" width="28" height="24" rx="4" fill={WH} stroke={BD} strokeWidth="2" />
            <rect x={prx} y={pry} width="12" height="12" rx="3" fill={BD} />
            <rect x={prx + 2} y={pry + 2} width="4" height="4" rx="2" fill={WH} />
            <rect x="68" y="104" width="44" height="10" rx="5" fill={BD} />
            <rect x="70" y="106" width="40" height="6" rx="3" fill={PK} opacity="0.55" />
          </g>
        </g>
      </svg>
    </div>
  );
}

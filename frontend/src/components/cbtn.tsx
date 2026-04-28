"use client";

import { useState } from "react";

type Props = {
  children: React.ReactNode;
  onClick?: () => void;
  accent?: string;
  textColor?: string;
  big?: boolean;
  disabled?: boolean;
  type?: "button" | "submit";
};

export function CBtn({
  children,
  onClick,
  accent = "#7c6fe0",
  textColor = "#fff",
  big = false,
  disabled = false,
  type = "button",
}: Props) {
  const [p, setP] = useState(false);
  return (
    <button
      type={type}
      onClick={onClick}
      onMouseDown={() => setP(true)}
      onMouseUp={() => setP(false)}
      onMouseLeave={() => setP(false)}
      disabled={disabled}
      style={{
        padding: big ? "15px 34px" : "11px 24px",
        borderRadius: 14,
        fontFamily: "inherit",
        border: "2.5px solid #1e1a3a",
        background: disabled ? "#cfcae6" : accent,
        color: textColor,
        fontWeight: 800,
        fontSize: big ? 15 : 13,
        cursor: disabled ? "not-allowed" : "pointer",
        boxShadow: p && !disabled ? "1px 1px 0 #1e1a3a" : "4px 4px 0 #1e1a3a",
        transform: p && !disabled ? "translate(3px,3px)" : "none",
        transition: "transform 0.08s, box-shadow 0.08s",
        opacity: disabled ? 0.7 : 1,
      }}
    >
      {children}
    </button>
  );
}

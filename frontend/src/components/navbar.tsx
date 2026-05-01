"use client";

import { useTheme } from "@/lib/theme";

export type PageId = "home" | "how" | "analyzer" | "jobs";

type Props = {
  page: PageId;
  navigate: (to: PageId) => void;
};

function ThemeToggle() {
  const { theme, toggle, mounted } = useTheme();
  const isDark = theme === "dark";
  // Pre-hydration: render a neutral, theme-independent placeholder so the
  // server HTML and the first client render are identical. Once mounted,
  // we know the real theme (set by the inline script in layout.tsx) and
  // can swap in the proper icon/colors.
  const baseStyle: React.CSSProperties = {
    width: 36,
    height: 36,
    marginLeft: 6,
    borderRadius: 10,
    border: "2.5px solid #1e1a3a",
    fontSize: 16,
    fontWeight: 800,
    cursor: "pointer",
    fontFamily: "inherit",
    boxShadow: "2px 2px 0 #1e1a3a",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "background 0.15s, color 0.15s",
  };

  if (!mounted) {
    return (
      <button
        aria-hidden
        tabIndex={-1}
        suppressHydrationWarning
        style={{ ...baseStyle, background: "transparent", color: "transparent", cursor: "default" }}
      >
        ☀️
      </button>
    );
  }

  return (
    <button
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      style={{
        ...baseStyle,
        background: isDark ? "#1e1a3a" : "#FDE68A",
        color: isDark ? "#FDE68A" : "#1e1a3a",
      }}
    >
      {isDark ? "🌙" : "☀️"}
    </button>
  );
}

export function Navbar({ page, navigate }: Props) {
  const pages: { id: PageId; label: string }[] = [
    { id: "home", label: "🏠 Home" },
    { id: "how", label: "📖 How It Works" },
    { id: "analyzer", label: "🔬 Analyzer" },
    { id: "jobs", label: "💼 Jobs" },
  ];
  return (
    <nav
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 200,
        background: "var(--pd-nav-bg)",
        backdropFilter: "blur(14px)",
        borderBottom: "2.5px solid #1e1a3a",
        boxShadow: "0 4px 0 #1e1a3a",
      }}
    >
      <div
        style={{
          maxWidth: 960,
          margin: "0 auto",
          padding: "0 24px",
          height: 64,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <button
          onClick={() => navigate("home")}
          style={{ display: "flex", alignItems: "center", gap: 10, background: "none", border: "none", cursor: "pointer", fontFamily: "inherit" }}
        >
          <div
            style={{
              width: 36,
              height: 36,
              background: "#0EA5E9",
              border: "2.5px solid #1e1a3a",
              borderRadius: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              boxShadow: "2px 2px 0 #1e1a3a",
            }}
          >
            🧭
          </div>
          <span style={{ fontWeight: 900, fontSize: 20, color: "var(--pd-text)", letterSpacing: "-0.02em" }}>
            Path<span style={{ color: "#0EA5E9" }}>Drift</span>
          </span>
        </button>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {pages.map((p) => (
            <button
              key={p.id}
              onClick={() => navigate(p.id)}
              style={{
                padding: "7px 14px",
                borderRadius: 10,
                border: page === p.id ? "2px solid #1e1a3a" : "2px solid transparent",
                background: page === p.id ? "#0EA5E9" : "transparent",
                color: page === p.id ? "#fff" : "var(--pd-text-muted)",
                fontWeight: 800,
                fontSize: 13,
                cursor: "pointer",
                fontFamily: "inherit",
                boxShadow: page === p.id ? "2px 2px 0 #1e1a3a" : "none",
                transform: page === p.id ? "translate(-1px,-1px)" : "none",
                transition: "all 0.15s",
              }}
            >
              {p.label}
            </button>
          ))}
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}

"use client";

export type PageId = "home" | "how" | "analyzer";

type Props = {
  page: PageId;
  navigate: (to: PageId) => void;
};

export function Navbar({ page, navigate }: Props) {
  const pages: { id: PageId; label: string }[] = [
    { id: "home", label: "🏠 Home" },
    { id: "how", label: "📖 How It Works" },
    { id: "analyzer", label: "🔬 Analyzer" },
  ];
  return (
    <nav
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 200,
        background: "rgba(255,255,255,0.92)",
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
          <span style={{ fontWeight: 900, fontSize: 20, color: "#1e1a3a", letterSpacing: "-0.02em" }}>
            Path<span style={{ color: "#0EA5E9" }}>Drift</span>
          </span>
        </button>
        <div style={{ display: "flex", gap: 6 }}>
          {pages.map((p) => (
            <button
              key={p.id}
              onClick={() => navigate(p.id)}
              style={{
                padding: "7px 14px",
                borderRadius: 10,
                border: page === p.id ? "2px solid #1e1a3a" : "2px solid transparent",
                background: page === p.id ? "#0EA5E9" : "transparent",
                color: page === p.id ? "#fff" : "#475569",
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
        </div>
      </div>
    </nav>
  );
}

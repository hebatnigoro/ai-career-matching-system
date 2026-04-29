export function BgBlobs() {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
      <div style={{ position: "absolute", top: "-8%", left: "-6%", width: 340, height: 340, borderRadius: "50%", background: "radial-gradient(circle, #FFD6A5 0%, transparent 68%)", opacity: 0.7 }} />
      <div style={{ position: "absolute", bottom: "-6%", right: "-4%", width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, #A8E6CF 0%, transparent 68%)", opacity: 0.6 }} />
      <div style={{ position: "absolute", top: "40%", right: "3%", width: 200, height: 200, borderRadius: "50%", background: "radial-gradient(circle, #FFB5C5 0%, transparent 68%)", opacity: 0.55 }} />
      <div style={{ position: "absolute", top: "12%", left: "55%", width: 180, height: 180, borderRadius: "50%", background: "radial-gradient(circle, #FFE26A 0%, transparent 68%)", opacity: 0.4 }} />
    </div>
  );
}

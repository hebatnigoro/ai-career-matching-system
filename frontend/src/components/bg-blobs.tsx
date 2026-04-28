export function BgBlobs() {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
      <div style={{ position: "absolute", top: "-8%", left: "-6%", width: 340, height: 340, borderRadius: "50%", background: "radial-gradient(circle, #d4ceff 0%, transparent 68%)", opacity: 0.65 }} />
      <div style={{ position: "absolute", bottom: "-6%", right: "-4%", width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, #bff0e8 0%, transparent 68%)", opacity: 0.55 }} />
      <div style={{ position: "absolute", top: "40%", right: "3%", width: 200, height: 200, borderRadius: "50%", background: "radial-gradient(circle, #ffd6e8 0%, transparent 68%)", opacity: 0.4 }} />
    </div>
  );
}

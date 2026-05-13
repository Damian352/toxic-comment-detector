import { useMemo, useState } from "react";

type PredictResponse = {
  probabilities: Record<string, number>;
  labels: string[];
};

const backendOrigin = import.meta.env.VITE_BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const sorted = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      setResult(body as PredictResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: 24 }}>
      <header style={{ marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 28 }}>Toxic Comment Detector</h1>
        <p style={{ margin: "8px 0 0", color: "#475569" }}>
          Multi-label probabilities for toxic, insult, threat, obscene, and related categories.
        </p>
      </header>

      <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>Comment text</label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 10,
          border: "1px solid #cbd5e1",
          fontSize: 15,
          resize: "vertical",
        }}
        placeholder="Paste a comment to analyze…"
      />

      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <button
          type="button"
          onClick={() => void analyze()}
          disabled={loading || text.trim().length === 0}
          style={{
            padding: "10px 16px",
            borderRadius: 10,
            border: "1px solid #0f172a",
            background: "#0f172a",
            color: "#fff",
            cursor: loading || text.trim().length === 0 ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
        <a href={`${backendOrigin}/docs`} target="_blank" rel="noreferrer" style={{ color: "#2563eb" }}>
          OpenAPI docs
        </a>
      </div>

      {error ? (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            borderRadius: 10,
            border: "1px solid #fecaca",
            background: "#fef2f2",
            color: "#991b1b",
            whiteSpace: "pre-wrap",
          }}
        >
          {error}
        </div>
      ) : null}

      {result ? (
        <section style={{ marginTop: 20 }}>
          <h2 style={{ margin: "0 0 12px", fontSize: 18 }}>Scores</h2>
          <div style={{ display: "grid", gap: 10 }}>
            {sorted.map(([label, p]) => (
              <div key={label} style={{ display: "grid", gap: 6 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span style={{ fontWeight: 600 }}>{label}</span>
                  <span style={{ color: "#475569" }}>{(p * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: 10, background: "#e2e8f0", borderRadius: 999 }}>
                  <div
                    style={{
                      height: 10,
                      width: `${Math.min(100, p * 100)}%`,
                      borderRadius: 999,
                      background: p >= 0.5 ? "#ef4444" : "#38bdf8",
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";

type ModelId = "tfidf_lr" | "bert" | "both";

type ModelInfo = {
  id: ModelId;
  name: string;
  description: string;
  loaded: boolean;
  artifact_path: string;
};

type ProjectionPoint = {
  id: string;
  text: string;
  labels: string[];
  x: number;
  y: number;
  similarity: number;
  is_active: boolean;
};

type PredictResponse = {
  probabilities: Record<string, number>;
  labels: string[];
  model: ModelId;
  similarity_projection: ProjectionPoint[];
  
  is_dual?: boolean;
  probabilities_tfidf?: Record<string, number>;
  probabilities_bert?: Record<string, number>;
  similarity_projection_tfidf?: ProjectionPoint[];
  similarity_projection_bert?: ProjectionPoint[];
};

type MetricLabelInfo = {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

type ModelMetrics = {
  hamming_loss: number;
  f1_macro: number;
  f1_micro: number;
  precision_macro: number;
  precision_micro: number;
  recall_macro: number;
  recall_micro: number;
  per_label: MetricLabelInfo[];
  dataset: {
    n_samples: number;
    n_train: number;
    n_test: number;
  };
};

type MetricsResponse = {
  tfidf_lr: ModelMetrics;
  bert: ModelMetrics;
};

const MODEL_LABELS: Record<ModelId, string> = {
  tfidf_lr: "TF-IDF + Logistic Regression",
  bert: "BERT (transformer)",
  both: "Dual Comparison Mode",
};

const LABEL_DESCRIPTIONS: Record<string, string> = {
  toxic: "General rude, aggressive, or offensive language demeaning others.",
  severe_toxic: "Extremely hateful, highly aggressive, or obscene abuse.",
  obscene: "Vulgarity, profanity, swearing, or sexually explicit terms.",
  threat: "Statements of intent to inflict bodily harm, violence, or death.",
  insult: "Demeaning remarks, name-calling, or target-specific degradation.",
  identity_hate: "Hate speech targeting race, gender, religion, sexual orientation, etc.",
};

const SAMPLES = [
  {
    label: "Clean & Positive",
    text: "This is a great and helpful project. Keep up the amazing work! Have a wonderful day.",
  },
  {
    label: "Toxic & Insult",
    text: "You are a complete idiot and should shut up right now! Nobody cares about your stupid opinion.",
  },
  {
    label: "Violence & Threat",
    text: "I will find you and hurt you, you better watch your back. This is not over.",
  },
  {
    label: "Multi-category Abuse",
    text: "Stupid piece of garbage, go back to where you came from! You make me sick.",
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<"analysis" | "metrics">("analysis");
  const [text, setText] = useState("");
  const [model, setModel] = useState<ModelId>("both"); // default to Dual Comparison Mode
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Metrics Dashboard State
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<"f1" | "precision" | "recall">("f1");

  // Interaction Tooltip States - Radar
  const [predictionHoveredIdx, setPredictionHoveredIdx] = useState<number | null>(null);
  const [predictionTooltip, setPredictionTooltip] = useState<{
    x: number;
    y: number;
    label: string;
    value: number;
    valueTfidf?: number;
    valueBert?: number;
  } | null>(null);

  // Interaction Tooltip States - Metrics
  const [metricsHoveredBar, setMetricsHoveredBar] = useState<{
    modelId: "tfidf_lr" | "bert";
    label: string;
    metric: string;
    value: number;
    support: number;
    x: number;
    y: number;
  } | null>(null);

  // Interaction Tooltip States - 2D Projection Space
  const [hoveredProjectionId, setHoveredProjectionId] = useState<string | null>(null);
  const [projectionTooltip, setProjectionTooltip] = useState<{
    x: number;
    y: number;
    point: ProjectionPoint;
    modelType?: "tfidf" | "bert";
  } | null>(null);

  // Nearest Neighbor Toggled Model in Dual Mode
  const [closestMatchesModel, setClosestMatchesModel] = useState<"tfidf" | "bert">("bert");

  // Load models on startup
  useEffect(() => {
    void fetch("/api/models")
      .then((res) => (res.ok ? res.json() : []))
      .then((data: ModelInfo[]) => {
        if (Array.isArray(data) && data.length > 0) {
          setModels(data);
          const firstLoaded = data.find((m) => m.loaded);
          if (firstLoaded) setModel(firstLoaded.id);
        }
      })
      .catch(() => {
        /* backend may be offline */
      });
  }, []);

  // Load metrics from backend
  useEffect(() => {
    setLoadingMetrics(true);
    fetch("/api/metrics")
      .then((res) => (res.ok ? res.json() : null))
      .then((data: MetricsResponse | null) => {
        if (data) setMetrics(data);
      })
      .catch((err) => {
        console.error("Failed to fetch metrics", err);
      })
      .finally(() => {
        setLoadingMetrics(false);
      });
  }, []);

  const sortedResult = useMemo(() => {
    if (!result) return [];
    // If dual, sort by the average or just by BERT probabilities (which is primary)
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const selectedModelInfo = models?.find((m) => m.id === model);

  // Extract closest similar comments from projection
  const closestMatches = useMemo(() => {
    if (!result) return [];
    if (result.is_dual) {
      const proj = closestMatchesModel === "bert" ? result.similarity_projection_bert : result.similarity_projection_tfidf;
      if (!proj) return [];
      return proj
        .filter((pt) => !pt.is_active)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3);
    } else {
      if (!result.similarity_projection) return [];
      return result.similarity_projection
        .filter((pt) => !pt.is_active)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3);
    }
  }, [result, closestMatchesModel]);

  async function analyze() {
    if (text.trim().length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
        if (res.status === 404) {
          throw new Error(
            "API not found. Start the backend: cd backend && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010\n" +
              "Then restart the frontend (npm run dev) so Vite picks up the proxy on port 8010.",
          );
        }
        throw new Error(detail || `Request failed (${res.status})`);
      }
      setResult(body as PredictResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // --- Radar Chart Computations ---
  const radarLabels = useMemo(() => {
    if (result) {
      return Object.keys(result.probabilities);
    }
    return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];
  }, [result]);

  const getRadarCoords = (index: number, value: number) => {
    const angle = (index * Math.PI) / 3 - Math.PI / 2; // subtract PI/2 to point upwards
    const center = 150;
    const maxRadius = 100;
    return {
      x: center + value * maxRadius * Math.cos(angle),
      y: center + value * maxRadius * Math.sin(angle),
    };
  };

  const radarPolygons = useMemo(() => {
    if (!result) return { predictionPath: "", points: [] };
    
    if (result.is_dual && result.probabilities_tfidf && result.probabilities_bert) {
      const pointsTfidf = radarLabels.map((label, idx) => {
        const p = result.probabilities_tfidf![label] ?? 0;
        return {
          ...getRadarCoords(idx, p),
          label,
          value: p,
          idx,
        };
      });
      const pointsBert = radarLabels.map((label, idx) => {
        const p = result.probabilities_bert![label] ?? 0;
        return {
          ...getRadarCoords(idx, p),
          label,
          value: p,
          idx,
        };
      });
      const pathTfidf = pointsTfidf.map((pt) => `${pt.x},${pt.y}`).join(" ");
      const pathBert = pointsBert.map((pt) => `${pt.x},${pt.y}`).join(" ");
      return {
        predictionPath: pathBert, // primary fallback
        points: pointsBert,       // primary fallback
        isDual: true,
        pathTfidf,
        pathBert,
        pointsTfidf,
        pointsBert,
      };
    } else {
      const points = radarLabels.map((label, idx) => {
        const p = result.probabilities[label] ?? 0;
        return {
          ...getRadarCoords(idx, p),
          label,
          value: p,
          idx,
        };
      });
      const path = points.map((pt) => `${pt.x},${pt.y}`).join(" ");
      return { predictionPath: path, points, isDual: false };
    }
  }, [result, radarLabels]);

  const handleRadarVertexHover = (
    e: React.MouseEvent<SVGCircleElement>,
    idx: number,
    label: string,
    val: number,
    valTfidf?: number,
    valBert?: number
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const parentRect = e.currentTarget.parentElement?.getBoundingClientRect();
    if (parentRect) {
      setPredictionTooltip({
        x: rect.left - parentRect.left + rect.width / 2,
        y: rect.top - parentRect.top - 8,
        label,
        value: val,
        valueTfidf: valTfidf,
        valueBert: valBert,
      });
      setPredictionHoveredIdx(idx);
    }
  };

  // --- 2D Space Projection Coordinate Mapping ---
  const getProjectionSVGCoords = (x: number, y: number) => {
    const center = 160;
    const scale = 1.35;
    return {
      cx: center + x * scale,
      cy: center - y * scale,
    };
  };

  const getPointColor = (pt: ProjectionPoint) => {
    if (pt.is_active) return "#eab308";
    const primaryLabel = pt.labels[0] || "safe";
    switch (primaryLabel) {
      case "safe":
        return "#10b981";
      case "threat":
      case "severe_toxic":
        return "#f97316";
      case "obscene":
        return "#ec4899";
      case "identity_hate":
        return "#a855f7";
      default:
        return "#ef4444";
    }
  };

  // --- Global Styles ---
  const styles = {
    container: {
      maxWidth: 960,
      margin: "0 auto",
      padding: "32px 16px",
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      color: "#1e293b",
      lineHeight: 1.5,
    },
    header: {
      marginBottom: 32,
      borderBottom: "1px solid #e2e8f0",
      paddingBottom: 20,
    },
    title: {
      margin: 0,
      fontSize: 32,
      fontWeight: 800,
      color: "#0f172a",
      letterSpacing: "-0.025em",
    },
    subtitle: {
      margin: "8px 0 0",
      color: "#64748b",
      fontSize: 16,
    },
    tabBar: {
      display: "flex",
      gap: 12,
      marginBottom: 24,
      borderBottom: "2px solid #e2e8f0",
      paddingBottom: 1,
    },
    tabButton: (isActive: boolean) => ({
      padding: "10px 16px",
      fontSize: 15,
      fontWeight: 600,
      backgroundColor: "transparent",
      border: "none",
      borderBottom: isActive ? "3px solid #3b82f6" : "3px solid transparent",
      color: isActive ? "#3b82f6" : "#64748b",
      cursor: "pointer",
      transition: "all 0.2s ease",
      marginBottom: -2,
    }),
    card: {
      backgroundColor: "#fff",
      borderRadius: 16,
      border: "1px solid #e2e8f0",
      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05)",
      padding: 24,
      marginBottom: 24,
    },
    sectionTitle: {
      margin: "0 0 16px",
      fontSize: 20,
      fontWeight: 700,
      color: "#0f172a",
    },
    label: {
      display: "block",
      fontWeight: 600,
      fontSize: 14,
      color: "#475569",
      marginBottom: 8,
      textTransform: "uppercase" as const,
      letterSpacing: "0.05em",
    },
    textarea: {
      width: "100%",
      padding: 16,
      borderRadius: 12,
      border: "1px solid #cbd5e1",
      fontSize: 16,
      lineHeight: 1.6,
      resize: "vertical" as const,
      boxSizing: "border-box" as const,
      outline: "none",
      transition: "border-color 0.2s ease, box-shadow 0.2s ease",
    },
    primaryButton: (disabled: boolean) => ({
      padding: "12px 24px",
      borderRadius: 10,
      backgroundColor: disabled ? "#94a3b8" : "#0f172a",
      color: "#fff",
      fontWeight: 600,
      fontSize: 15,
      border: "none",
      cursor: disabled ? "not-allowed" : "pointer",
      transition: "all 0.2s ease",
    }),
    sampleButton: {
      padding: "6px 12px",
      borderRadius: 20,
      backgroundColor: "#f1f5f9",
      border: "1px solid #e2e8f0",
      color: "#475569",
      fontSize: 13,
      fontWeight: 500,
      cursor: "pointer",
      transition: "all 0.15s ease",
    },
    gridTwoCols: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
      gap: 24,
    },
    probabilityRow: (isPositive: boolean) => ({
      display: "flex",
      flexDirection: "column" as const,
      gap: 6,
      padding: "10px 12px",
      borderRadius: 8,
      backgroundColor: isPositive ? "#fff5f5" : "#f8fafc",
      border: `1px solid ${isPositive ? "#fee2e2" : "#f1f5f9"}`,
      transition: "all 0.2s ease",
    }),
    metricToggle: (isActive: boolean) => ({
      padding: "8px 16px",
      borderRadius: 8,
      fontSize: 14,
      fontWeight: 600,
      backgroundColor: isActive ? "#2563eb" : "#f1f5f9",
      color: isActive ? "#fff" : "#475569",
      border: "none",
      cursor: "pointer",
      transition: "all 0.2s ease",
    }),
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <h1 style={styles.title}>Toxic Comment Detector</h1>
          <span
            style={{
              padding: "4px 8px",
              backgroundColor: "#dbeafe",
              color: "#1e40af",
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            v0.2.0
          </span>
        </div>
        <p style={styles.subtitle}>
          Multi-label classification of natural language comments using TF-IDF and fine-tuned Transformer (BERT) models.
        </p>
      </header>

      {/* Tab Navigation */}
      <div style={styles.tabBar}>
        <button
          style={styles.tabButton(activeTab === "analysis")}
          onClick={() => setActiveTab("analysis")}
        >
          🔮 Analyze Text
        </button>
        <button
          style={styles.tabButton(activeTab === "metrics")}
          onClick={() => setActiveTab("metrics")}
        >
          📊 Model Comparison & Metrics
        </button>
      </div>

      {/* Tab 1: Analysis */}
      {activeTab === "analysis" && (
        <div>
          <div style={styles.gridTwoCols}>
            {/* Input Form Column */}
            <div>
              <div style={styles.card}>
                <h2 style={styles.sectionTitle}>Model Configuration</h2>
                <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 20 }}>
                  {(models ?? (Object.keys(MODEL_LABELS) as ModelId[]).map((id) => ({ id, name: MODEL_LABELS[id], loaded: true, description: "", artifact_path: "" }))).map(
                    (m) => (
                      <label
                        key={m.id}
                        style={{
                          display: "flex",
                          gap: 12,
                          alignItems: "flex-start",
                          padding: 12,
                          borderRadius: 10,
                          border: `2px solid ${model === m.id ? "#3b82f6" : "#e2e8f0"}`,
                          backgroundColor: model === m.id ? "#eff6ff" : "transparent",
                          cursor: m.loaded === false ? "not-allowed" : "pointer",
                          opacity: m.loaded === false ? 0.5 : 1,
                          transition: "all 0.2s ease",
                        }}
                      >
                        <input
                          type="radio"
                          name="model"
                          value={m.id}
                          checked={model === m.id}
                          disabled={m.loaded === false}
                          onChange={() => setModel(m.id)}
                          style={{ marginTop: 4, width: 16, height: 16 }}
                        />
                        <div>
                          <span style={{ fontWeight: 700, fontSize: 15, color: model === m.id ? "#1d4ed8" : "#1e293b" }}>
                            {m.name ?? MODEL_LABELS[m.id]}
                          </span>
                          {m.loaded === false ? (
                            <span style={{ color: "#b45309", marginLeft: 8, fontSize: 13, fontWeight: 600 }}>
                              (not loaded)
                            </span>
                          ) : null}
                          <span style={{ display: "block", color: "#64748b", fontSize: 13, marginTop: 4 }}>
                            {m.description || (
                              m.id === "tfidf_lr" ? "Word and character n-grams with One-vs-Rest Logistic Regression" : 
                              m.id === "bert" ? "Fine-tuned BERT with context-aware representations" :
                              "Run both models side-by-side to visually inspect the difference in predictions"
                            )}
                          </span>
                        </div>
                      </label>
                    )
                  )}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <label style={styles.label}>Comment text</label>
                  <span style={{ fontSize: 12, color: "#64748b" }}>{text.length} / 8000 chars</span>
                </div>

                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={6}
                  style={styles.textarea}
                  placeholder="Paste a comment to analyze or click a sample below…"
                />

                {/* Samples */}
                <div style={{ marginTop: 12, marginBottom: 20 }}>
                  <span style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6, fontWeight: 600 }}>
                    💡 QUICK TEST SAMPLES:
                  </span>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {SAMPLES.map((sample, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => {
                          setText(sample.text);
                          setResult(null);
                        }}
                        style={styles.sampleButton}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#e2e8f0")}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#f1f5f9")}
                      >
                        {sample.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <button
                    type="button"
                    onClick={() => void analyze()}
                    disabled={loading || text.trim().length === 0 || selectedModelInfo?.loaded === false}
                    style={styles.primaryButton(loading || text.trim().length === 0 || selectedModelInfo?.loaded === false)}
                  >
                    {loading ? "Analyzing text..." : "🚀 Analyze Toxicity"}
                  </button>
                  {error && (
                    <span style={{ color: "#ef4444", fontSize: 14, fontWeight: 500 }}>
                      ⚠️ Analysis failed
                    </span>
                  )}
                </div>
              </div>

              {error && (
                <div
                  style={{
                    padding: 16,
                    borderRadius: 12,
                    border: "1px solid #fecaca",
                    background: "#fef2f2",
                    color: "#991b1b",
                    fontSize: 14,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {error}
                </div>
              )}
            </div>

            {/* Visualizations and Results Column */}
            <div>
              <div style={styles.card}>
                <h2 style={styles.sectionTitle}>Analysis Output & Visualizations</h2>
                {!result && !loading && (
                  <div style={{ textAlign: "center", padding: "48px 16px", color: "#94a3b8" }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>🕵️‍♂️</div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 16 }}>No Analysis Loaded</p>
                    <p style={{ margin: "4px 0 0", fontSize: 14 }}>
                      Input or select a comment and click "Analyze Toxicity" to visualize probabilities.
                    </p>
                  </div>
                )}

                {loading && (
                  <div style={{ textAlign: "center", padding: "64px 16px", color: "#64748b" }}>
                    <div style={{ fontSize: 32, marginBottom: 12, animation: "spin 1s linear infinite" }} className="animate-spin">
                      ⏳
                    </div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 16 }}>Processing text...</p>
                    <p style={{ margin: "4px 0 0", fontSize: 14 }}>
                      Running {MODEL_LABELS[model]} inference
                    </p>
                  </div>
                )}

                {result && (
                  <div style={{ position: "relative" }}>
                    <div
                      style={{
                        padding: "8px 12px",
                        backgroundColor: "#f8fafc",
                        borderRadius: 8,
                        marginBottom: 16,
                        borderLeft: "4px solid #3b82f6",
                        fontSize: 13,
                        color: "#475569",
                        display: "flex",
                        justifyContent: "space-between",
                      }}
                    >
                      <span>Model Mode: <strong>{MODEL_LABELS[result.model]}</strong></span>
                      <span>Max score: <strong>{(Math.max(...Object.values(result.probabilities)) * 100).toFixed(1)}%</strong></span>
                    </div>

                    {/* Radar Chart Legend if Dual Mode */}
                    {result.is_dual && (
                      <div style={{ display: "flex", justifyContent: "center", gap: 16, marginBottom: 10 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 12, height: 12, backgroundColor: "#3b82f6", borderRadius: "50%" }} />
                          <span style={{ fontSize: 12, fontWeight: "bold", color: "#3b82f6" }}>TF-IDF + LR</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 12, height: 12, backgroundColor: "#8b5cf6", borderRadius: "50%" }} />
                          <span style={{ fontSize: 12, fontWeight: "bold", color: "#8b5cf6" }}>BERT (Transformer)</span>
                        </div>
                      </div>
                    )}

                    {/* Interactive Radar Chart Container */}
                    <div style={{ display: "flex", justifyContent: "center", marginBottom: 24, position: "relative" }}>
                      <svg width="300" height="300" style={{ overflow: "visible" }}>
                        {/* Background Polygons for Scale levels */}
                        {[0.2, 0.4, 0.6, 0.8, 1.0].map((level) => {
                          const levelPoints = [0, 1, 2, 3, 4, 5]
                            .map((j) => {
                              const coords = getRadarCoords(j, level);
                              return `${coords.x},${coords.y}`;
                            })
                            .join(" ");
                          return (
                            <g key={level}>
                              <polygon
                                points={levelPoints}
                                fill="none"
                                stroke="#e2e8f0"
                                strokeWidth="1"
                                strokeDasharray="4 2"
                              />
                              {/* Scale text label */}
                              <text
                                x="150"
                                y={150 - level * 100 + 4}
                                fill="#94a3b8"
                                fontSize="9"
                                textAnchor="middle"
                                fontWeight="bold"
                              >
                                {`${level * 100}%`}
                              </text>
                            </g>
                          );
                        })}

                        {/* Radial Axis Lines */}
                        {[0, 1, 2, 3, 4, 5].map((idx) => {
                          const end = getRadarCoords(idx, 1.0);
                          return (
                            <line
                              key={idx}
                              x1="150"
                              y1="150"
                              x2={end.x}
                              y2={end.y}
                              stroke="#cbd5e1"
                              strokeWidth="1.5"
                            />
                          );
                        })}

                        {/* Label Placements */}
                        {radarLabels.map((label, idx) => {
                          const textCoords = getRadarCoords(idx, 1.22);
                          const textAnchor = idx === 0 || idx === 3 ? "middle" : idx === 1 || idx === 2 ? "start" : "end";
                          const isHigh = result.is_dual 
                            ? (result.probabilities_bert?.[label] ?? 0) >= 0.5 || (result.probabilities_tfidf?.[label] ?? 0) >= 0.5
                            : (result.probabilities[label] ?? 0) >= 0.5;
                          return (
                            <text
                              key={label}
                              x={textCoords.x}
                              y={textCoords.y + 4}
                              textAnchor={textAnchor}
                              fill={isHigh ? "#ef4444" : "#475569"}
                              fontSize="11"
                              fontWeight={isHigh ? "bold" : "600"}
                              style={{ transition: "all 0.2s ease" }}
                            >
                              {label}
                            </text>
                          );
                        })}

                        {/* Polygons for TF-IDF & BERT in Dual Mode */}
                        {radarPolygons.isDual ? (
                          <>
                            {/* TF-IDF Polygon */}
                            <polygon
                              points={radarPolygons.pathTfidf}
                              fill="#3b82f622"
                              stroke="#3b82f6"
                              strokeWidth="2"
                              style={{ transition: "all 0.3s ease" }}
                            />
                            {/* BERT Polygon */}
                            <polygon
                              points={radarPolygons.pathBert}
                              fill="#8b5cf622"
                              stroke="#8b5cf6"
                              strokeWidth="2.5"
                              style={{ transition: "all 0.3s ease" }}
                            />
                          </>
                        ) : (
                          /* Standard Single Model Polygon */
                          radarPolygons.predictionPath && (
                            <polygon
                              points={radarPolygons.predictionPath}
                              fill={Math.max(...Object.values(result.probabilities)) >= 0.5 ? "#fca5a577" : "#7dd3fc77"}
                              stroke={Math.max(...Object.values(result.probabilities)) >= 0.5 ? "#ef4444" : "#0284c7"}
                              strokeWidth="2.5"
                              style={{ transition: "all 0.3s ease" }}
                            />
                          )
                        )}

                        {/* Dots for TF-IDF in Dual Mode */}
                        {radarPolygons.isDual && radarPolygons.pointsTfidf?.map((pt) => {
                          const isHovered = predictionHoveredIdx === pt.idx;
                          return (
                            <circle
                              key={`tfidf_${pt.label}`}
                              cx={pt.x}
                              cy={pt.y}
                              r={isHovered ? 7 : 4}
                              fill="#3b82f6"
                              stroke="#fff"
                              strokeWidth="1.5"
                              style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                              onMouseEnter={(e) => handleRadarVertexHover(
                                e, 
                                pt.idx, 
                                pt.label, 
                                pt.value, 
                                pt.value, 
                                result.probabilities_bert?.[pt.label] ?? 0
                              )}
                              onMouseLeave={() => {
                                setPredictionTooltip(null);
                                setPredictionHoveredIdx(null);
                              }}
                            />
                          );
                        })}

                        {/* Dots for BERT / Main Model */}
                        {radarPolygons.isDual ? (
                          radarPolygons.pointsBert?.map((pt) => {
                            const isHovered = predictionHoveredIdx === pt.idx;
                            return (
                              <circle
                                key={`bert_${pt.label}`}
                                cx={pt.x}
                                cy={pt.y}
                                r={isHovered ? 8 : 5}
                                fill="#8b5cf6"
                                stroke="#fff"
                                strokeWidth="2"
                                style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                                onMouseEnter={(e) => handleRadarVertexHover(
                                  e, 
                                  pt.idx, 
                                  pt.label, 
                                  pt.value, 
                                  result.probabilities_tfidf?.[pt.label] ?? 0, 
                                  pt.value
                                )}
                                onMouseLeave={() => {
                                  setPredictionTooltip(null);
                                  setPredictionHoveredIdx(null);
                                }}
                              />
                            );
                          })
                        ) : (
                          /* Standard Single Dots */
                          radarPolygons.points.map((pt) => {
                            const isHovered = predictionHoveredIdx === pt.idx;
                            const isHigh = pt.value >= 0.5;
                            return (
                              <circle
                                key={pt.label}
                                cx={pt.x}
                                cy={pt.y}
                                r={isHovered ? 8 : 5}
                                fill={isHigh ? "#ef4444" : "#0284c7"}
                                stroke="#fff"
                                strokeWidth="2"
                                style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                                onMouseEnter={(e) => handleRadarVertexHover(e, pt.idx, pt.label, pt.value)}
                                onMouseLeave={() => {
                                  setPredictionTooltip(null);
                                  setPredictionHoveredIdx(null);
                                }}
                              />
                            );
                          })
                        )}
                      </svg>

                      {/* Tooltip Popup on Hover (Customized for Dual Mode comparison) */}
                      {predictionTooltip && (
                        <div
                          style={{
                            position: "absolute",
                            left: predictionTooltip.x,
                            top: predictionTooltip.y - (predictionTooltip.valueTfidf !== undefined ? 130 : 105),
                            transform: "translateX(-50%)",
                            width: 240,
                            padding: 12,
                            backgroundColor: "#1e293b",
                            color: "#fff",
                            borderRadius: 8,
                            fontSize: 12,
                            boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
                            zIndex: 50,
                            pointerEvents: "none",
                            lineHeight: 1.4,
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #475569", paddingBottom: 4, marginBottom: 6 }}>
                            <span style={{ fontWeight: 800, textTransform: "uppercase" }}>{predictionTooltip.label}</span>
                          </div>

                          {predictionTooltip.valueTfidf !== undefined && predictionTooltip.valueBert !== undefined ? (
                            // Dual Comparison Tooltip Info
                            <div style={{ display: "grid", gap: 4, marginBottom: 6 }}>
                              <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "#60a5fa", fontWeight: 600 }}>🔵 TF-IDF + LR:</span>
                                <strong style={{ color: predictionTooltip.valueTfidf >= 0.5 ? "#fca5a5" : "#60a5fa" }}>
                                  {(predictionTooltip.valueTfidf * 100).toFixed(1)}%
                                </strong>
                              </div>
                              <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "#c084fc", fontWeight: 600 }}>🟣 BERT (Transformer):</span>
                                <strong style={{ color: predictionTooltip.valueBert >= 0.5 ? "#fca5a5" : "#c084fc" }}>
                                  {(predictionTooltip.valueBert * 100).toFixed(1)}%
                                </strong>
                              </div>
                            </div>
                          ) : (
                            // Single Model Tooltip Info
                            <>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span>Score probability:</span>
                                <strong style={{ color: predictionTooltip.value >= 0.5 ? "#fca5a5" : "#38bdf8" }}>
                                  {(predictionTooltip.value * 100).toFixed(1)}%
                                </strong>
                              </div>
                              <div style={{ marginBottom: 6 }}>
                                <strong>Status: </strong>
                                <span style={{ color: predictionTooltip.value >= 0.5 ? "#ef4444" : "#10b981", fontWeight: "bold" }}>
                                  {predictionTooltip.value >= 0.5 ? "🔴 TOXIC" : "🟢 SAFE"}
                                </span>
                              </div>
                            </>
                          )}
                          <p style={{ margin: 0, color: "#cbd5e1", fontSize: 11 }}>
                            {LABEL_DESCRIPTIONS[predictionTooltip.label] ?? ""}
                          </p>
                        </div>
                      )}
                    </div>

                    <div style={{ textAlign: "center", fontSize: 12, color: "#64748b", margin: "-12px 0 20px" }}>
                      💡 <em>Hover over vertex points on the radar web to inspect side-by-side model predictions.</em>
                    </div>

                    {/* Scores list */}
                    <div style={{ display: "grid", gap: 10 }}>
                      <label style={styles.label}>Scores Breakdown & Model Contrast</label>
                      {sortedResult.map(([label]) => {
                        const labelIdx = radarLabels.indexOf(label);
                        const isHovered = predictionHoveredIdx === labelIdx;

                        if (result.is_dual && result.probabilities_tfidf && result.probabilities_bert) {
                          const pTfidf = result.probabilities_tfidf[label] ?? 0;
                          const pBert = result.probabilities_bert[label] ?? 0;
                          const isHigh = pTfidf >= 0.5 || pBert >= 0.5;

                          return (
                            <div
                              key={label}
                              style={{
                                ...styles.probabilityRow(isHigh),
                                boxShadow: isHovered ? "0 0 8px rgba(139, 92, 246, 0.2)" : "none",
                                borderColor: isHovered ? "#8b5cf6" : isHigh ? "#fee2e2" : "#f1f5f9",
                              }}
                              onMouseEnter={() => setPredictionHoveredIdx(labelIdx)}
                              onMouseLeave={() => setPredictionHoveredIdx(null)}
                            >
                              <span style={{ fontWeight: 700, fontSize: 14, textTransform: "capitalize", marginBottom: 2 }}>
                                {label.replace("_", " ")}
                              </span>

                              {/* Progress bar for TF-IDF */}
                              <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 40px", gap: 8, alignItems: "center" }}>
                                <span style={{ fontSize: 11, color: "#475569" }}>🔵 TF-IDF + LR</span>
                                <div style={{ height: 6, background: "#e2e8f0", borderRadius: 999, overflow: "hidden" }}>
                                  <div
                                    style={{
                                      height: "100%",
                                      width: `${pTfidf * 100}%`,
                                      borderRadius: 999,
                                      backgroundColor: pTfidf >= 0.5 ? "#f87171" : "#3b82f6",
                                    }}
                                  />
                                </div>
                                <span style={{ fontSize: 11, fontWeight: "bold", textAlign: "right" }}>{(pTfidf * 100).toFixed(1)}%</span>
                              </div>

                              {/* Progress bar for BERT */}
                              <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 40px", gap: 8, alignItems: "center" }}>
                                <span style={{ fontSize: 11, color: "#475569" }}>🟣 BERT</span>
                                <div style={{ height: 6, background: "#e2e8f0", borderRadius: 999, overflow: "hidden" }}>
                                  <div
                                    style={{
                                      height: "100%",
                                      width: `${pBert * 100}%`,
                                      borderRadius: 999,
                                      backgroundColor: pBert >= 0.5 ? "#ef4444" : "#8b5cf6",
                                    }}
                                  />
                                </div>
                                <span style={{ fontSize: 11, fontWeight: "bold", textAlign: "right" }}>{(pBert * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          );
                        } else {
                          // Standard Single progress bar
                          const p = result.probabilities[label] ?? 0;
                          const isHigh = p >= 0.5;
                          return (
                            <div
                              key={label}
                              style={{
                                ...styles.probabilityRow(isHigh),
                                boxShadow: isHovered ? "0 0 8px rgba(59, 130, 246, 0.2)" : "none",
                                borderColor: isHovered ? "#3b82f6" : isHigh ? "#fee2e2" : "#f1f5f9",
                              }}
                              onMouseEnter={() => setPredictionHoveredIdx(labelIdx)}
                              onMouseLeave={() => setPredictionHoveredIdx(null)}
                            >
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <span style={{ fontWeight: 700, fontSize: 15, textTransform: "capitalize" }}>
                                  {label.replace("_", " ")}
                                </span>
                                <span style={{ fontWeight: "bold", color: isHigh ? "#ef4444" : "#475569", fontSize: 15 }}>
                                  {(p * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div style={{ height: 8, background: "#e2e8f0", borderRadius: 999, overflow: "hidden" }}>
                                <div
                                  style={{
                                    height: "100%",
                                    width: `${p * 100}%`,
                                    borderRadius: 999,
                                    background: isHigh
                                      ? "linear-gradient(90deg, #f87171, #ef4444)"
                                      : "linear-gradient(90deg, #38bdf8, #0284c7)",
                                    transition: "width 0.6s ease",
                                  }}
                                />
                              </div>
                              <p style={{ margin: "2px 0 0", fontSize: 11, color: "#64748b" }}>
                                {LABEL_DESCRIPTIONS[label]}
                              </p>
                            </div>
                          );
                        }
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* SECTION: 2D Embedding Projection & Closest Matches */}
          {result && (
            <div style={{ ...styles.card, marginTop: 24 }}>
              <div style={{ borderBottom: "1px solid #e2e8f0", paddingBottom: 12, marginBottom: 20 }}>
                <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                  🌐 Semantic Space Mapping & Nearest Neighbors
                </h3>
                <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: 14 }}>
                  Compare where this comment lies in 2D space relative to typical clean and toxic benchmarks in our corpus.
                </p>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
                {/* 2D Projection Chart Column */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <span style={styles.label}>
                    {result.is_dual ? "Dual Model Vector Alignment" : "2D Vector Projection Space"}
                  </span>
                  <div style={{ position: "relative", width: 320, height: 320, border: "1px solid #cbd5e1", borderRadius: 12, backgroundColor: "#f8fafc", overflow: "visible" }}>
                    
                    {/* SVG Canvas for scatter plot */}
                    <svg width="320" height="320" style={{ overflow: "visible" }}>
                      {/* Quadrant backgrounds/axis */}
                      <line x1="160" y1="0" x2="160" y2="320" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="0" y1="160" x2="320" y2="160" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="3 3" />

                      {/* Quadrant labels */}
                      <text x="20" y="30" fill="#94a3b8" fontSize="9" fontWeight="bold">🛡️ Clean Discussion</text>
                      <text x="300" y="30" fill="#94a3b8" fontSize="9" fontWeight="bold" textAnchor="end">⚠️ Threats & Hate</text>
                      <text x="300" y="300" fill="#94a3b8" fontSize="9" fontWeight="bold" textAnchor="end">🤬 Insults & Obscenity</text>

                      {/* Connection Line in Dual Mode */}
                      {result.is_dual && result.similarity_projection_tfidf && result.similarity_projection_bert && (() => {
                        const tfActive = result.similarity_projection_tfidf.find(p => p.is_active);
                        const bertActive = result.similarity_projection_bert.find(p => p.is_active);
                        if (tfActive && bertActive) {
                          const coordTf = getProjectionSVGCoords(tfActive.x, tfActive.y);
                          const coordBert = getProjectionSVGCoords(bertActive.x, bertActive.y);
                          return (
                            <line
                              x1={coordTf.cx}
                              y1={coordTf.cy}
                              x2={coordBert.cx}
                              y2={coordBert.cy}
                              stroke="#eab308"
                              strokeWidth="2"
                              strokeDasharray="4 4"
                            />
                          );
                        }
                        return null;
                      })()}

                      {/* Render corpus anchor reference points (using BERT's projection points as anchors) */}
                      {(result.is_dual ? result.similarity_projection_bert : result.similarity_projection)?.map((pt) => {
                        if (pt.is_active) return null; // handle active user dots separately below
                        
                        const { cx, cy } = getProjectionSVGCoords(pt.x, pt.y);
                        const isHovered = hoveredProjectionId === pt.id;
                        const color = getPointColor(pt);

                        return (
                          <circle
                            key={pt.id}
                            cx={cx}
                            cy={cy}
                            r={isHovered ? 9 : 5.5}
                            fill={color}
                            stroke="#fff"
                            strokeWidth={isHovered ? 2 : 1.5}
                            opacity={hoveredProjectionId && !isHovered ? 0.35 : 0.85}
                            style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                            onMouseEnter={(e) => {
                              setHoveredProjectionId(pt.id);
                              const rect = e.currentTarget.getBoundingClientRect();
                              const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                              if (parentRect) {
                                setProjectionTooltip({
                                  x: rect.left - parentRect.left + rect.width / 2,
                                  y: rect.top - parentRect.top - 8,
                                  point: pt,
                                });
                              }
                            }}
                            onMouseLeave={() => {
                              setHoveredProjectionId(null);
                              setProjectionTooltip(null);
                            }}
                          />
                        );
                      })}

                      {/* Render ACTIVE User Dot(s) */}
                      {result.is_dual ? (
                        <>
                          {/* 1. TF-IDF Active user dot */}
                          {(() => {
                            const tfActive = result.similarity_projection_tfidf?.find(p => p.is_active);
                            if (!tfActive) return null;
                            const { cx, cy } = getProjectionSVGCoords(tfActive.x, tfActive.y);
                            return (
                              <g key="active_user_tfidf">
                                <circle cx={cx} cy={cy} r="12" fill="none" stroke="#3b82f6" strokeWidth="1.5" opacity="0.5">
                                  <animate attributeName="r" values="6;14;6" dur="2.5s" repeatCount="indefinite" />
                                </circle>
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r="7"
                                  fill="#3b82f6"
                                  stroke="#fff"
                                  strokeWidth="2"
                                  style={{ cursor: "pointer" }}
                                  onMouseEnter={(e) => {
                                    setHoveredProjectionId("active_user_tfidf");
                                    const rect = e.currentTarget.getBoundingClientRect();
                                    const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                                    if (parentRect) {
                                      setProjectionTooltip({
                                        x: rect.left - parentRect.left + rect.width / 2,
                                        y: rect.top - parentRect.top - 8,
                                        point: tfActive,
                                        modelType: "tfidf"
                                      });
                                    }
                                  }}
                                  onMouseLeave={() => {
                                    setHoveredProjectionId(null);
                                    setProjectionTooltip(null);
                                  }}
                                />
                              </g>
                            );
                          })()}

                          {/* 2. BERT Active user dot */}
                          {(() => {
                            const bertActive = result.similarity_projection_bert?.find(p => p.is_active);
                            if (!bertActive) return null;
                            const { cx, cy } = getProjectionSVGCoords(bertActive.x, bertActive.y);
                            return (
                              <g key="active_user_bert">
                                <circle cx={cx} cy={cy} r="14" fill="none" stroke="#8b5cf6" strokeWidth="1.5" opacity="0.6">
                                  <animate attributeName="r" values="7;16;7" dur="2s" repeatCount="indefinite" />
                                </circle>
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r="8"
                                  fill="#8b5cf6"
                                  stroke="#fff"
                                  strokeWidth="2"
                                  style={{ cursor: "pointer" }}
                                  onMouseEnter={(e) => {
                                    setHoveredProjectionId("active_user_bert");
                                    const rect = e.currentTarget.getBoundingClientRect();
                                    const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                                    if (parentRect) {
                                      setProjectionTooltip({
                                        x: rect.left - parentRect.left + rect.width / 2,
                                        y: rect.top - parentRect.top - 8,
                                        point: bertActive,
                                        modelType: "bert"
                                      });
                                    }
                                  }}
                                  onMouseLeave={() => {
                                    setHoveredProjectionId(null);
                                    setProjectionTooltip(null);
                                  }}
                                />
                              </g>
                            );
                          })()}
                        </>
                      ) : (
                        /* Standard Single Active user dot */
                        (() => {
                          const activePt = result.similarity_projection?.find(p => p.is_active);
                          if (!activePt) return null;
                          const { cx, cy } = getProjectionSVGCoords(activePt.x, activePt.y);
                          return (
                            <g key="active_user_single">
                              <circle cx={cx} cy={cy} r="16" fill="none" stroke="#eab308" strokeWidth="2" opacity="0.4">
                                <animate attributeName="r" values="8;18;8" dur="2s" repeatCount="indefinite" />
                                <animate attributeName="opacity" values="0.7;0.1;0.7" dur="2s" repeatCount="indefinite" />
                              </circle>
                              <circle
                                cx={cx}
                                cy={cy}
                                r="8"
                                fill="#eab308"
                                stroke="#fff"
                                strokeWidth="2.5"
                                style={{ cursor: "pointer" }}
                                onMouseEnter={(e) => {
                                  setHoveredProjectionId(activePt.id);
                                  const rect = e.currentTarget.getBoundingClientRect();
                                  const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                                  if (parentRect) {
                                    setProjectionTooltip({
                                      x: rect.left - parentRect.left + rect.width / 2,
                                      y: rect.top - parentRect.top - 8,
                                      point: activePt,
                                    });
                                  }
                                }}
                                onMouseLeave={() => {
                                  setHoveredProjectionId(null);
                                  setProjectionTooltip(null);
                                }}
                              />
                            </g>
                          );
                        })()
                      )}
                    </svg>

                    {/* 2D Space Tooltip Popup */}
                    {projectionTooltip && (
                      <div
                        style={{
                          position: "absolute",
                          left: projectionTooltip.x,
                          top: projectionTooltip.y - 125,
                          transform: "translateX(-50%)",
                          width: 250,
                          padding: 12,
                          backgroundColor: "#0f172a",
                          color: "#fff",
                          borderRadius: 8,
                          fontSize: 12,
                          boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.4)",
                          zIndex: 60,
                          pointerEvents: "none",
                          lineHeight: 1.4,
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #334155", paddingBottom: 4, marginBottom: 6 }}>
                          <span style={{ fontWeight: 800, color: projectionTooltip.point.is_active ? "#eab308" : "#94a3b8" }}>
                            {projectionTooltip.point.is_active 
                              ? (projectionTooltip.modelType === "tfidf" ? "🔵 ACTIVE (TF-IDF)" : projectionTooltip.modelType === "bert" ? "🟣 ACTIVE (BERT)" : "⭐ ACTIVE TEXT") 
                              : "📌 REFERENCE"
                            }
                          </span>
                          {!projectionTooltip.point.is_active && (
                            <span style={{ fontWeight: 800, color: "#60a5fa" }}>
                              Match: {(projectionTooltip.point.similarity * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        <p style={{ margin: "0 0 6px", fontStyle: "italic", color: "#e2e8f0", fontSize: 11 }}>
                          "{projectionTooltip.point.text.length > 90 ? `${projectionTooltip.point.text.substring(0, 87)}...` : projectionTooltip.point.text}"
                        </p>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                          {projectionTooltip.point.labels.map((lbl: string) => (
                            <span
                              key={lbl}
                              style={{
                                padding: "2px 6px",
                                borderRadius: 4,
                                fontSize: 9,
                                fontWeight: "bold",
                                textTransform: "uppercase",
                                backgroundColor: lbl === "safe" ? "#064e3b" : "#7f1d1d",
                                color: lbl === "safe" ? "#a7f3d0" : "#fca5a5",
                              }}
                            >
                              {lbl.replace("_", " ")}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Legend */}
                  <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10, marginTop: 12, fontSize: 11, fontWeight: 600 }}>
                    {result.is_dual ? (
                      <>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#3b82f6" }} />
                          <span>You (TF-IDF)</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#8b5cf6" }} />
                          <span>You (BERT)</span>
                        </div>
                      </>
                    ) : (
                      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#eab308" }} />
                        <span>You</span>
                      </div>
                    )}
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#10b981" }} />
                      <span>Clean</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#ef4444" }} />
                      <span>Toxic</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#f97316" }} />
                      <span>Threat</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#ec4899" }} />
                      <span>Obscene</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: "#a855f7" }} />
                      <span>Hate Speech</span>
                    </div>
                  </div>
                </div>

                {/* Closest Matches Card List Column */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <span style={styles.label}>Closest Reference Comments</span>
                    
                    {/* Toggler in Dual Mode */}
                    {result.is_dual && (
                      <div style={{ display: "flex", gap: 4, backgroundColor: "#f1f5f9", padding: 2, borderRadius: 6 }}>
                        <button
                          style={{
                            padding: "3px 8px",
                            fontSize: 11,
                            fontWeight: "bold",
                            border: "none",
                            borderRadius: 4,
                            cursor: "pointer",
                            backgroundColor: closestMatchesModel === "tfidf" ? "#3b82f6" : "transparent",
                            color: closestMatchesModel === "tfidf" ? "#fff" : "#475569",
                          }}
                          onClick={() => setClosestMatchesModel("tfidf")}
                        >
                          TF-IDF
                        </button>
                        <button
                          style={{
                            padding: "3px 8px",
                            fontSize: 11,
                            fontWeight: "bold",
                            border: "none",
                            borderRadius: 4,
                            cursor: "pointer",
                            backgroundColor: closestMatchesModel === "bert" ? "#8b5cf6" : "transparent",
                            color: closestMatchesModel === "bert" ? "#fff" : "#475569",
                          }}
                          onClick={() => setClosestMatchesModel("bert")}
                        >
                          BERT
                        </button>
                      </div>
                    )}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {closestMatches.map((match) => {
                      const isHovered = hoveredProjectionId === match.id;
                      return (
                        <div
                          key={match.id}
                          style={{
                            padding: 14,
                            borderRadius: 12,
                            border: `2px solid ${isHovered ? (closestMatchesModel === "bert" && result.is_dual ? "#8b5cf6" : "#3b82f6") : "#e2e8f0"}`,
                            backgroundColor: isHovered ? "#f0f7ff" : "#fff",
                            transition: "all 0.15s ease",
                            cursor: "pointer",
                            position: "relative",
                          }}
                          onMouseEnter={() => setHoveredProjectionId(match.id)}
                          onMouseLeave={() => setHoveredProjectionId(null)}
                          onClick={() => {
                            setText(match.text);
                            setResult(null);
                          }}
                          title="Click to load this comment into the analyzer!"
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: closestMatchesModel === "bert" && result.is_dual ? "#7c3aed" : "#2563eb" }}>
                              🔥 Similarity Match: {(match.similarity * 100).toFixed(1)}%
                            </span>
                            <span style={{ fontSize: 11, color: "#64748b", textDecoration: "underline" }}>
                              Click to test 🚀
                            </span>
                          </div>
                          
                          <p style={{ margin: "0 0 10px", fontSize: 13, color: "#334155", fontStyle: "italic", lineHeight: 1.4 }}>
                            "{match.text}"
                          </p>

                          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                            {match.labels.map((lbl) => (
                              <span
                                key={lbl}
                                style={{
                                  padding: "3px 8px",
                                  borderRadius: 6,
                                  fontSize: 10,
                                  fontWeight: 700,
                                  textTransform: "uppercase",
                                  backgroundColor: lbl === "safe" ? "#d1fae5" : "#fee2e2",
                                  color: lbl === "safe" ? "#065f46" : "#991b1b",
                                }}
                              >
                                {lbl.replace("_", " ")}
                              </span>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ textAlign: "center", fontSize: 12, color: "#64748b", marginTop: 12 }}>
                    💡 <em>Clicking any reference card above copies it to the analyzer, helping you observe how different models respond to it.</em>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Metrics Evaluation */}
      {activeTab === "metrics" && (
        <div>
          {loadingMetrics && (
            <div style={{ textAlign: "center", padding: "64px 16px", color: "#64748b" }}>
              ⏳ Loading model metrics...
            </div>
          )}

          {!loadingMetrics && !metrics && (
            <div style={{ textAlign: "center", padding: "48px 16px", color: "#ef4444" }}>
              ⚠️ Failed to load model metrics from backend. Make sure the backend is running.
            </div>
          )}

          {metrics && (
            <div>
              {/* Overview Performance Cards */}
              <div style={styles.gridTwoCols}>
                <div
                  style={{
                    ...styles.card,
                    borderLeft: "6px solid #3b82f6",
                    background: "linear-gradient(135deg, #ffffff, #eff6ff)",
                  }}
                >
                  <h3 style={{ margin: "0 0 12px", fontSize: 18, fontWeight: 700, color: "#1e3050" }}>
                    🔵 TF-IDF + Logistic Regression
                  </h3>
                  <p style={{ fontSize: 14, color: "#64748b", margin: "0 0 16px" }}>
                    Baseline classical machine learning pipeline. Super fast training and inference, low resource consumption.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MACRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#1e3a8a" }}>
                        {(metrics.tfidf_lr.f1_macro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MICRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#1e3a8a" }}>
                        {(metrics.tfidf_lr.f1_micro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>HAMMING LOSS</span>
                      <strong style={{ fontSize: 20, color: "#1e3a8a" }}>
                        {metrics.tfidf_lr.hamming_loss.toFixed(4)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>TEST SAMPLES</span>
                      <strong style={{ fontSize: 20, color: "#1e3a8a" }}>
                        {metrics.tfidf_lr.dataset.n_test.toLocaleString()}
                      </strong>
                    </div>
                  </div>
                </div>

                <div
                  style={{
                    ...styles.card,
                    borderLeft: "6px solid #8b5cf6",
                    background: "linear-gradient(135deg, #ffffff, #faf5ff)",
                  }}
                >
                  <h3 style={{ margin: "0 0 12px", fontSize: 18, fontWeight: 700, color: "#311042" }}>
                    🟣 BERT (Transformer) Model
                  </h3>
                  <p style={{ fontSize: 14, color: "#64748b", margin: "0 0 16px" }}>
                    Fine-tuned context-aware encoder. Slower inference but superior language understanding and higher accuracy.
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MACRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#4c1d95" }}>
                        {(metrics.bert.f1_macro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MICRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#4c1d95" }}>
                        {(metrics.bert.f1_micro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>HAMMING LOSS</span>
                      <strong style={{ fontSize: 20, color: "#4c1d95" }}>
                        {metrics.bert.hamming_loss.toFixed(4)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>TEST SAMPLES</span>
                      <strong style={{ fontSize: 20, color: "#4c1d95" }}>
                        {metrics.bert.dataset.n_test.toLocaleString()}
                      </strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* Grouped Bar Chart of Class Metrics */}
              <div style={styles.card}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 20 }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>Class-by-Class Metric Comparison</h3>
                    <p style={{ margin: "2px 0 0", color: "#64748b", fontSize: 13 }}>
                      Compare precision, recall, and F1-score across Jigsaw toxicity classes.
                    </p>
                  </div>
                  {/* Metric Toggle */}
                  <div style={{ display: "flex", gap: 6, backgroundColor: "#f1f5f9", padding: 4, borderRadius: 10 }}>
                    <button
                      style={styles.metricToggle(selectedMetric === "f1")}
                      onClick={() => setSelectedMetric("f1")}
                    >
                      F1-Score
                    </button>
                    <button
                      style={styles.metricToggle(selectedMetric === "precision")}
                      onClick={() => setSelectedMetric("precision")}
                    >
                      Precision
                    </button>
                    <button
                      style={styles.metricToggle(selectedMetric === "recall")}
                      onClick={() => setSelectedMetric("recall")}
                    >
                      Recall
                    </button>
                  </div>
                </div>

                {/* SVG Grouped Bar Chart */}
                <div style={{ display: "flex", justifyContent: "center", position: "relative" }}>
                  <svg width="640" height="280" style={{ overflow: "visible" }}>
                    {/* Y-axis baseline grids */}
                    {[0.2, 0.4, 0.6, 0.8, 1.0].map((v) => (
                      <g key={v}>
                        <line
                          x1="50"
                          y1={240 - v * 200}
                          x2="630"
                          y2={240 - v * 200}
                          stroke="#e2e8f0"
                          strokeWidth="1"
                        />
                        <text x="40" y={240 - v * 200 + 4} textAnchor="end" fill="#94a3b8" fontSize="10">
                          {`${(v * 100).toFixed(0)}%`}
                        </text>
                      </g>
                    ))}
                    {/* Zero baseline */}
                    <line x1="50" y1="240" x2="630" y2="240" stroke="#cbd5e1" strokeWidth="1.5" />
                    <text x="40" y="244" textAnchor="end" fill="#94a3b8" fontSize="10">0%</text>

                    {/* Chart columns loop */}
                    {metrics.tfidf_lr.per_label.map((tfidfInfo, idx) => {
                      const label = tfidfInfo.label;
                      const bertInfo = metrics.bert.per_label.find((l) => l.label === label) || tfidfInfo;

                      // Extract value
                      const tfidfVal = tfidfInfo[selectedMetric];
                      const bertVal = bertInfo[selectedMetric];

                      // Bar calculations
                      const groupWidth = 90;
                      const groupX = 60 + idx * groupWidth;

                      const tfidfBarHeight = tfidfVal * 200;
                      const tfidfBarY = 240 - tfidfBarHeight;

                      const bertBarHeight = bertVal * 200;
                      const bertBarY = 240 - bertBarHeight;

                      // Check if hovered
                      const isTfidfHovered = metricsHoveredBar?.modelId === "tfidf_lr" && metricsHoveredBar?.label === label;
                      const isBertHovered = metricsHoveredBar?.modelId === "bert" && metricsHoveredBar?.label === label;

                      return (
                        <g key={label}>
                          {/* Label bottom text */}
                          <text
                            x={groupX + 32}
                            y="256"
                            textAnchor="middle"
                            fill="#475569"
                            fontSize="10"
                            fontWeight="bold"
                          >
                            {label.replace("_", " ")}
                          </text>

                          {/* TF-IDF Bar */}
                          <rect
                            x={groupX + 4}
                            y={tfidfBarY}
                            width="24"
                            height={tfidfBarHeight}
                            fill={isTfidfHovered ? "#2563eb" : "#3b82f6"}
                            rx="4"
                            style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                            onMouseEnter={(e) => {
                              const rect = e.currentTarget.getBoundingClientRect();
                              const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                              if (parentRect) {
                                setMetricsHoveredBar({
                                  modelId: "tfidf_lr",
                                  label,
                                  metric: selectedMetric,
                                  value: tfidfVal,
                                  support: tfidfInfo.support,
                                  x: rect.left - parentRect.left + rect.width / 2,
                                  y: rect.top - parentRect.top - 8,
                                });
                              }
                            }}
                            onMouseLeave={() => setMetricsHoveredBar(null)}
                          />

                          {/* BERT Bar */}
                          <rect
                            x={groupX + 34}
                            y={bertBarY}
                            width="24"
                            height={bertBarHeight}
                            fill={isBertHovered ? "#7c3aed" : "#8b5cf6"}
                            rx="4"
                            style={{ cursor: "pointer", transition: "all 0.15s ease" }}
                            onMouseEnter={(e) => {
                              const rect = e.currentTarget.getBoundingClientRect();
                              const parentRect = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect();
                              if (parentRect) {
                                setMetricsHoveredBar({
                                  modelId: "bert",
                                  label,
                                  metric: selectedMetric,
                                  value: bertVal,
                                  support: bertInfo.support,
                                  x: rect.left - parentRect.left + rect.width / 2,
                                  y: rect.top - parentRect.top - 8,
                                });
                              }
                            }}
                            onMouseLeave={() => setMetricsHoveredBar(null)}
                          />
                        </g>
                      );
                    })}
                  </svg>

                  {/* Tooltip for metrics chart */}
                  {metricsHoveredBar && (
                    <div
                      style={{
                        position: "absolute",
                        left: metricsHoveredBar.x,
                        top: metricsHoveredBar.y - 100,
                        transform: "translateX(-50%)",
                        width: 200,
                        padding: 12,
                        backgroundColor: "#1e293b",
                        color: "#fff",
                        borderRadius: 8,
                        fontSize: 12,
                        boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
                        zIndex: 50,
                        pointerEvents: "none",
                        lineHeight: 1.4,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #475569", paddingBottom: 4, marginBottom: 6 }}>
                        <span style={{ fontWeight: 800, textTransform: "uppercase" }}>{metricsHoveredBar.label}</span>
                        <span style={{ fontWeight: 800, color: metricsHoveredBar.modelId === "bert" ? "#c084fc" : "#60a5fa" }}>
                          {(metricsHoveredBar.value * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ marginBottom: 4 }}>
                        <strong>Model: </strong>
                        {metricsHoveredBar.modelId === "bert" ? "BERT" : "TF-IDF + LR"}
                      </div>
                      <div style={{ marginBottom: 4 }}>
                        <strong>Metric: </strong>
                        <span style={{ textTransform: "capitalize" }}>{metricsHoveredBar.metric}</span>
                      </div>
                      <div>
                        <strong>Test Support: </strong>
                        {metricsHoveredBar.support.toLocaleString()} samples
                      </div>
                    </div>
                  )}
                </div>

                {/* Legend */}
                <div style={{ display: "flex", justifyContent: "center", gap: 24, marginTop: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 14, height: 14, backgroundColor: "#3b82f6", borderRadius: 3 }} />
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#475569" }}>TF-IDF + Logistic Regression</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 14, height: 14, backgroundColor: "#8b5cf6", borderRadius: 3 }} />
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#475569" }}>BERT (Fine-tuned Transformer)</span>
                  </div>
                </div>
              </div>

              {/* Analysis and Insights */}
              <div style={styles.card}>
                <h3 style={{ margin: "0 0 12px", fontSize: 18, fontWeight: 700 }}>🔍 Key Evaluation Insights</h3>
                <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10, fontSize: 14, color: "#475569" }}>
                  <li>
                    <strong>Transformer Superiority:</strong> BERT outperforms the classical TF-IDF baseline significantly across almost all classes, achieving a Macro F1-score of <strong>67.9%</strong> compared to TF-IDF's <strong>61.6%</strong>.
                  </li>
                  <li>
                    <strong>Rare Class Performance:</strong> Underrepresented categories like <code>threat</code> (97 test samples) and <code>identity_hate</code> (289 test samples) see the largest relative performance jumps with BERT (e.g. <code>threat</code> F1 goes from <strong>45.0%</strong> to <strong>53.3%</strong>, and <code>identity_hate</code> goes from <strong>46.9%</strong> to <strong>61.0%</strong>) due to contextual understanding.
                  </li>
                  <li>
                    <strong>Precision vs. Recall Trade-off:</strong> Classical TF-IDF has very high Recall (e.g. <strong>87.3%</strong> on <code>toxic</code>) because <code>class_weight="balanced"</code> bias was applied to reduce False Negatives, but at the cost of lower Precision (e.g. <strong>71.7%</strong> on <code>toxic</code>). BERT achieves a much healthier balance, boosting Precision substantially (e.g. <code>toxic</code> Precision is <strong>84.8%</strong>) while maintaining superb Recall (<strong>86.2%</strong>).
                  </li>
                  <li>
                    <strong>Hamming Loss Reduction:</strong> BERT reduces the overall multilabel prediction error (Hamming Loss) by nearly 40%, dropping from <strong>0.0232</strong> down to <strong>0.0143</strong>.
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import ProjectionPlot3D from "./components/ProjectionPlot3D";

type ModelId = "tfidf_lr" | "bert" | "both";

type ModelInfo = {
  id: ModelId;
  name: string;
  description: string;
  loaded: boolean;
  artifact_path: string;
};

type ProjectionErrorType =
  | "correct"
  | "false_positive"
  | "false_negative"
  | "label_mismatch"
  | null;

type ProjectionPoint = {
  id: string;
  text: string;
  labels: string[];
  x: number;
  y: number;
  z?: number;
  similarity: number;
  is_active: boolean;
  is_validation?: boolean;
  ground_truth_labels?: string[];
  predicted_labels?: string[];
  error_type?: ProjectionErrorType;
};

type AnalysisLang = "auto" | "en" | "pl";

type PredictResponse = {
  probabilities: Record<string, number>;
  labels: string[];
  model: ModelId;
  similarity_projection: ProjectionPoint[];

  requested_lang?: AnalysisLang;
  analysis_lang?: "en" | "pl";
  lang_confidence?: number | null;
  lang_source?: string | null;

  is_dual?: boolean;
  probabilities_tfidf?: Record<string, number>;
  probabilities_bert?: Record<string, number>;
  similarity_projection_tfidf?: ProjectionPoint[];
  similarity_projection_bert?: ProjectionPoint[];
  projection_method?: string | null;
  projection_axes?: Record<string, string> | null;
  explained_variance_ratio?: number[] | null;

  reference_projection?: ProjectionPoint[];
  reference_projection_tfidf?: ProjectionPoint[];
  reference_projection_bert?: ProjectionPoint[];

  pca_included?: boolean;
};

type MapVisualizationMode = "pca" | "anchors";
type ProjectionErrorFilter = "all" | "correct" | "errors" | "false_positive" | "false_negative" | "label_mismatch";

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
  tfidf_lr_pl: ModelMetrics;
  bert_pl: ModelMetrics;
};

function isMetricsResponse(data: unknown): data is MetricsResponse {
  if (!data || typeof data !== "object") return false;
  const record = data as Record<string, unknown>;
  const required = ["tfidf_lr", "bert", "tfidf_lr_pl", "bert_pl"] as const;
  return required.every((key) => {
    const block = record[key];
    return (
      !!block &&
      typeof block === "object" &&
      typeof (block as ModelMetrics).f1_macro === "number" &&
      Array.isArray((block as ModelMetrics).per_label)
    );
  });
}

async function fetchMetricsFromApi(): Promise<MetricsResponse | null> {
  const res = await fetch("/api/metrics");
  if (!res.ok) return null;
  const data: unknown = await res.json();
  return isMetricsResponse(data) ? data : null;
}

async function fetchMetricsWithRetry(attempts = 3, delayMs = 1500): Promise<MetricsResponse | null> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const data = await fetchMetricsFromApi();
    if (data) return data;
    if (attempt < attempts - 1) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  return null;
}

const MODEL_LABELS: Record<ModelId, string> = {
  tfidf_lr: "TF-IDF + Logistic Regression",
  bert: "BERT (transformer)",
  both: "Dual Comparison Mode",
};

const LABEL_DESCRIPTIONS: Record<string, string> = {
  // English Labels
  toxic: "General rude, aggressive, or offensive language demeaning others.",
  severe_toxic: "Extremely hateful, highly aggressive, or obscene abuse.",
  obscene: "Vulgarity, profanity, swearing, or sexually explicit terms.",
  threat: "Statements of intent to inflict bodily harm, violence, or death.",
  insult: "Demeaning remarks, name-calling, or target-specific degradation.",
  identity_hate: "Hate speech targeting race, gender, religion, sexual orientation, etc.",
  
  // Polish Labels
  safe: "Bezpieczny, neutralny lub pozytywny komentarz niezawierający naruszeń.",
  hate_speech: "Mowa nienawiści, atakowanie osób lub grup na tle rasowym, religijnym, narodowościowym itp.",
  violence: "Promowanie przemocy, drastyczne opisy, nawoływanie do agresji lub samobójstwa.",
  vulgarity: "Wulgaryzmy, rynsztokowe słownictwo, zniesławienie lub naruszenie dóbr osobistych.",
};

const EN_SAMPLES = [
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

const DEFAULT_THRESHOLD = 0.5;

type RowVariant = "violation" | "safe" | "neutral";

function isLabelActive(label: string, probability: number, lang: "en" | "pl", threshold = DEFAULT_THRESHOLD): boolean {
  return probability >= threshold;
}

function isCommentSafe(probs: Record<string, number>, lang: "en" | "pl", threshold = DEFAULT_THRESHOLD): boolean {
  if (lang === "pl") {
    const hasViolation = Object.entries(probs).some(([label, p]) => label !== "safe" && p >= threshold);
    if (hasViolation) return false;
    return (probs.safe ?? 0) >= threshold || !hasViolation;
  }
  return !Object.values(probs).some((p) => p >= threshold);
}

function getLabelRowVariant(
  label: string,
  lang: "en" | "pl",
  ...probs: number[]
): RowVariant {
  const active = probs.some((p) => isLabelActive(label, p, lang));
  if (!active) return "neutral";
  if (label === "safe") return "safe";
  return "violation";
}

function getBarColor(label: string, probability: number, lang: "en" | "pl", modelHue: string): string {
  if (!isLabelActive(label, probability, lang)) return modelHue;
  if (label === "safe") return "#10b981";
  return "#ef4444";
}

const PL_SAMPLES = [
  {
    label: "Pozytywny (Safe)",
    text: "Dziękuję bardzo za pomoc! Świetna robota, miłego dnia życzę wszystkim!",
  },
  {
    label: "Obraźliwy (Hate Speech)",
    text: "Ty kompletny idioto, zamknij się wreszcie i nie pisz tych głupot na forum.",
  },
  {
    label: "Agresywny (Violence)",
    text: "Znajdę cię i połamię ci nogi gnoju, pożałujesz tego bardzo szybko, zobaczysz.",
  },
  {
    label: "Rynsztokowy (Vulgarity)",
    text: "Wyp***dalaj stąd ty głupi ch***u! Co ty p***dolisz w ogóle za bzdury?!",
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<"analysis" | "metrics">("analysis");
  const [uiLang, setUiLang] = useState<"en" | "pl">("en");
  const [analysisLang, setAnalysisLang] = useState<AnalysisLang>("auto");
  const [text, setText] = useState("");
  const [model, setModel] = useState<ModelId>("both");
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Metrics Dashboard State
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
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

  // Scatter map controls
  const [includePca, setIncludePca] = useState(false);
  const [mapVisualizationMode, setMapVisualizationMode] = useState<MapVisualizationMode>("anchors");
  const [projectionDim, setProjectionDim] = useState<2 | 3>(2);
  const [errorFilter, setErrorFilter] = useState<ProjectionErrorFilter>("all");

  const effectiveAnalysisLang = useMemo((): "en" | "pl" => {
    if (result?.analysis_lang === "en" || result?.analysis_lang === "pl") {
      return result.analysis_lang;
    }
    return analysisLang === "auto" ? "en" : analysisLang;
  }, [result, analysisLang]);

  const modelsLangParam = analysisLang === "auto" ? "en" : analysisLang;

  // Load models on startup and when analysis language changes
  useEffect(() => {
    void fetch(`/api/models?lang=${modelsLangParam}`)
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
  }, [modelsLangParam]);

  const loadMetrics = async (force = false) => {
    if (loadingMetrics) return;
    if (metrics && !force) return;
    setLoadingMetrics(true);
    setMetricsError(null);
    try {
      const data = await fetchMetricsWithRetry();
      if (data) {
        setMetrics(data);
      } else {
        setMetricsError(
          uiLang === "pl"
            ? "Nie udało się pobrać statystyk z API."
            : "Could not fetch metrics from the API.",
        );
      }
    } catch (err) {
      console.error("Failed to fetch metrics", err);
      setMetricsError(
        uiLang === "pl"
          ? "Błąd połączenia podczas pobierania statystyk."
          : "Connection error while fetching metrics.",
      );
    } finally {
      setLoadingMetrics(false);
    }
  };

  useEffect(() => {
    void loadMetrics();
  }, []);

  useEffect(() => {
    if (activeTab === "metrics" && !metrics && !loadingMetrics) {
      void loadMetrics(true);
    }
  }, [activeTab]);

  const handleAnalysisLangChange = (newLang: AnalysisLang) => {
    setAnalysisLang(newLang);
    setResult(null);
    setText("");
    setError(null);
  };

  const activeSamples = useMemo(() => {
    return effectiveAnalysisLang === "pl" ? PL_SAMPLES : EN_SAMPLES;
  }, [effectiveAnalysisLang]);

  const activeMetrics = useMemo(() => {
    if (!metrics) return null;
    return effectiveAnalysisLang === "pl"
      ? { tfidf_lr: metrics.tfidf_lr_pl, bert: metrics.bert_pl }
      : { tfidf_lr: metrics.tfidf_lr, bert: metrics.bert };
  }, [metrics, effectiveAnalysisLang]);

  const sortedResult = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const commentIsSafe = useMemo(() => {
    if (!result) return null;
    const projection = result.is_dual ? result.similarity_projection_bert : result.similarity_projection;
    const activePoint = projection?.find((pt) => pt.is_active);
    if (activePoint?.labels?.length) {
      return activePoint.labels.length === 1 && activePoint.labels[0] === "safe";
    }
    return isCommentSafe(result.probabilities, effectiveAnalysisLang);
  }, [result, effectiveAnalysisLang]);

  const selectedModelInfo = models?.find((m) => m.id === model);

  // Extract closest similar comments from projection
  const activeProjectionSource = useMemo(() => {
    if (!result) return [] as ProjectionPoint[];
    const useAnchors = mapVisualizationMode === "anchors";
    if (result.is_dual) {
      if (useAnchors) {
        return closestMatchesModel === "bert"
          ? result.reference_projection_bert ?? []
          : result.reference_projection_tfidf ?? [];
      }
      return closestMatchesModel === "bert"
        ? result.similarity_projection_bert ?? []
        : result.similarity_projection_tfidf ?? [];
    }
    if (useAnchors) return result.reference_projection ?? [];
    return result.similarity_projection ?? [];
  }, [result, closestMatchesModel, mapVisualizationMode]);

  const closestMatches = useMemo(() => {
    if (!activeProjectionSource.length) return [];
    return activeProjectionSource
      .filter((pt) => !pt.is_active)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3);
  }, [activeProjectionSource]);

  const primaryProjectionPoints = activeProjectionSource;

  const displayedProjectionPoints = useMemo(() => {
    return primaryProjectionPoints.filter((pt) => {
      if (pt.is_active) return true;
      if (mapVisualizationMode === "anchors") return true;
      if (errorFilter === "all") return true;
      if (errorFilter === "errors") return pt.error_type != null && pt.error_type !== "correct";
      return pt.error_type === errorFilter;
    });
  }, [primaryProjectionPoints, errorFilter, mapVisualizationMode]);

  const axisLabelX = result?.projection_axes?.x ?? "PC1";
  const axisLabelY = result?.projection_axes?.y ?? "PC2";
  const axisLabelZ = result?.projection_axes?.z ?? "PC3";

  async function analyze() {
    if (text.trim().length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model, lang: analysisLang, include_pca: includePca }),
      });
      
      let body: any = {};
      let parseFailed = false;
      try {
        const textData = await res.text();
        try {
          body = JSON.parse(textData);
        } catch {
          body = { detail: textData };
          parseFailed = true;
        }
      } catch {
        body = {};
      }

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
      const parsed = body as PredictResponse;
      setResult(parsed);
      if (!parsed.pca_included) {
        setMapVisualizationMode("anchors");
      }
      if (!metrics) {
        void loadMetrics(true);
      }
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
    return effectiveAnalysisLang === "pl"
      ? ["safe", "hate_speech", "violence", "vulgarity"]
      : ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];
  }, [result, effectiveAnalysisLang]);

  const numVertices = radarLabels.length;

  const getRadarCoords = (index: number, value: number) => {
    const angle = (index * Math.PI * 2) / numVertices - Math.PI / 2; // split circle into dynamic sectors
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
        predictionPath: pathBert,
        points: pointsBert,
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
  }, [result, radarLabels, numVertices]);

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

  // --- PCA projection coordinate mapping ---
  const getDisplayCoords = (pt: ProjectionPoint): { x: number; y: number } => ({
    x: pt.x,
    y: pt.y,
  });

  const getProjectionSVGCoords = (x: number, y: number) => {
    const center = 160;
    const scale = 1.35;
    return {
      cx: center + x * scale,
      cy: center - y * scale,
    };
  };

  const isProjectionPointSafe = (pt: ProjectionPoint) =>
    pt.labels.length === 1 && pt.labels[0] === "safe";

  const getActiveUserColor = (pt: ProjectionPoint) =>
    isProjectionPointSafe(pt) ? "#10b981" : "#ef4444";

  const getErrorTypeColor = (errorType: ProjectionErrorType | undefined) => {
    switch (errorType) {
      case "false_positive":
        return "#ef4444";
      case "false_negative":
        return "#f97316";
      case "label_mismatch":
        return "#a855f7";
      case "correct":
        return "#94a3b8";
      default:
        return "#64748b";
    }
  };

  const getPointColor = (pt: ProjectionPoint) => {
    if (pt.is_active) return getActiveUserColor(pt);
    if (pt.is_validation && pt.error_type) return getErrorTypeColor(pt.error_type);
    const primaryLabel = pt.labels[0] || "safe";
    switch (primaryLabel) {
      case "safe":
        return "#10b981";
      case "threat":
      case "severe_toxic":
      case "violence":
        return "#f97316";
      case "obscene":
      case "vulgarity":
        return "#ec4899";
      case "identity_hate":
      case "hate_speech":
        return "#a855f7";
      default:
        return "#ef4444";
    }
  };

  const plot3DColor = (pt: ProjectionPoint) => getPointColor(pt);

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
      marginBottom: 24,
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
    probabilityRow: (variant: RowVariant) => ({
      display: "flex",
      flexDirection: "column" as const,
      gap: 6,
      padding: "10px 12px",
      borderRadius: 8,
      backgroundColor: variant === "violation" ? "#fff5f5" : variant === "safe" ? "#f0fdf4" : "#f8fafc",
      border: `1px solid ${variant === "violation" ? "#fee2e2" : variant === "safe" ? "#bbf7d0" : "#f1f5f9"}`,
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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
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
          
          {/* Beautiful Separate Language Selectors */}
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            {/* Interface Language Selector */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {uiLang === "pl" ? "Język interfejsu" : "Interface Language"}
              </span>
              <div style={{ display: "flex", background: "#f1f5f9", padding: 3, borderRadius: 999, border: "1px solid #cbd5e1" }}>
                <button
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    backgroundColor: uiLang === "en" ? "#0f172a" : "transparent",
                    color: uiLang === "en" ? "#fff" : "#475569",
                  }}
                  onClick={() => setUiLang("en")}
                >
                  🇺🇸 EN
                </button>
                <button
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    backgroundColor: uiLang === "pl" ? "#0f172a" : "transparent",
                    color: uiLang === "pl" ? "#fff" : "#475569",
                  }}
                  onClick={() => setUiLang("pl")}
                >
                  🇵🇱 PL
                </button>
              </div>
            </div>

            {/* Comment/Analysis Language Selector */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                {uiLang === "pl" ? "Język analizy" : "Analysis Language"}
              </span>
              <div style={{ display: "flex", background: "#f1f5f9", padding: 3, borderRadius: 999, border: "1px solid #cbd5e1" }}>
                <button
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    backgroundColor: analysisLang === "auto" ? "#2563eb" : "transparent",
                    color: analysisLang === "auto" ? "#fff" : "#475569",
                  }}
                  onClick={() => handleAnalysisLangChange("auto")}
                >
                  {uiLang === "pl" ? "🔍 Auto" : "🔍 Auto"}
                </button>
                <button
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    backgroundColor: analysisLang === "en" ? "#2563eb" : "transparent",
                    color: analysisLang === "en" ? "#fff" : "#475569",
                  }}
                  onClick={() => handleAnalysisLangChange("en")}
                >
                  🇺🇸 English (Jigsaw)
                </button>
                <button
                  style={{
                    padding: "6px 12px",
                    borderRadius: 999,
                    fontSize: 12,
                    fontWeight: 700,
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    backgroundColor: analysisLang === "pl" ? "#2563eb" : "transparent",
                    color: analysisLang === "pl" ? "#fff" : "#475569",
                  }}
                  onClick={() => handleAnalysisLangChange("pl")}
                >
                  🇵🇱 Polski (BAN-PL)
                </button>
              </div>
            </div>
          </div>
        </div>
        <p style={styles.subtitle}>
          Multi-label classification of comments. Supporting English (Jigsaw) and Polish (BAN-PL) datasets with dual TF-IDF and BERT models.
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
                <h2 style={styles.sectionTitle}>
                  {uiLang === "pl" ? "Konfiguracja Modelu" : "Model Configuration"} (
                  {analysisLang === "auto"
                    ? uiLang === "pl"
                      ? "AUTO"
                      : "AUTO"
                    : analysisLang.toUpperCase()}
                  )
                </h2>
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
                              ({uiLang === "pl" ? "niezaładowany" : "not loaded"})
                            </span>
                          ) : null}
                          <span style={{ display: "block", color: "#64748b", fontSize: 13, marginTop: 4 }}>
                            {m.description || (
                              m.id === "tfidf_lr" ? (uiLang === "pl" ? "Cechy n-gramów słów i znaków z regresją logistyczną One-vs-Rest" : "Word and character n-grams with One-vs-Rest Logistic Regression") : 
                              m.id === "bert" ? (effectiveAnalysisLang === "pl" ? (uiLang === "pl" ? "Dostrojony HerBERT dla polskiego kontekstu" : "Fine-tuned HerBERT for Polish context") : (uiLang === "pl" ? "Dostrojony model BERT reprezentacji kontekstowych" : "Fine-tuned BERT with context-aware representations")) :
                              (uiLang === "pl" ? "Uruchom oba modele obok siebie, aby porównać ich prognozy" : "Run both models side-by-side to visually inspect the difference in predictions")
                            )}
                          </span>
                        </div>
                      </label>
                    )
                  )}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <label style={styles.label}>
                    {uiLang === "pl" ? "Treść komentarza" : "Comment text"} (
                    {analysisLang === "auto"
                      ? uiLang === "pl"
                        ? "język wykrywany automatycznie"
                        : "language auto-detected"
                      : analysisLang === "en"
                        ? "English"
                        : "Polish"}
                    )
                  </label>
                  <span style={{ fontSize: 12, color: "#64748b" }}>{text.length} / 8000 chars</span>
                </div>

                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={6}
                  style={styles.textarea}
                  placeholder={
                    analysisLang === "auto"
                      ? uiLang === "pl"
                        ? "Wpisz komentarz w dowolnym języku (EN/PL) — język zostanie wykryty automatycznie..."
                        : "Paste a comment in English or Polish — language will be detected automatically..."
                      : analysisLang === "pl"
                        ? uiLang === "pl"
                          ? "Wpisz polski komentarz do analizy..."
                          : "Paste a Polish comment to analyze..."
                        : uiLang === "pl"
                          ? "Wpisz angielski komentarz do analizy..."
                          : "Paste an English comment to analyze or click a sample below…"
                  }
                />

                {/* Samples */}
                <div style={{ marginTop: 12, marginBottom: 20 }}>
                  <span style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6, fontWeight: 600 }}>
                    {uiLang === "pl" ? "💡 PRZYKŁADOWE TEKSTY:" : "💡 QUICK TEST SAMPLES:"}
                  </span>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {activeSamples.map((sample, idx) => (
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

                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 12,
                    fontSize: 13,
                    color: "#475569",
                    cursor: "pointer",
                    userSelect: "none",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={includePca}
                    onChange={(e) => setIncludePca(e.target.checked)}
                    style={{ width: 16, height: 16, accentColor: "#4338ca" }}
                  />
                  <span>
                    {uiLang === "pl"
                      ? "Dołącz mapę PCA (wolniejsze — dodatkowe embeddingi BERT/TF-IDF)"
                      : "Include PCA validation map (slower — extra BERT/TF-IDF embeddings)"}
                  </span>
                </label>

                <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <button
                    type="button"
                    onClick={() => void analyze()}
                    disabled={loading || text.trim().length === 0 || selectedModelInfo?.loaded === false}
                    style={styles.primaryButton(loading || text.trim().length === 0 || selectedModelInfo?.loaded === false)}
                  >
                    {loading ? (uiLang === "pl" ? "Analizowanie..." : "Analyzing text...") : (uiLang === "pl" ? "🚀 Analizuj komentarz" : "🚀 Analyze Toxicity")}
                  </button>
                  {error && (
                    <span style={{ color: "#ef4444", fontSize: 14, fontWeight: 500 }}>
                      ⚠️ {uiLang === "pl" ? "Analiza nieudana" : "Analysis failed"}
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
                <h2 style={styles.sectionTitle}>{uiLang === "pl" ? "Wizualizacja wyników analizy" : "Analysis Output & Visualizations"}</h2>
                {!result && !loading && (
                  <div style={{ textAlign: "center", padding: "48px 16px", color: "#94a3b8" }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>🕵️‍♂️</div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 16 }}>{uiLang === "pl" ? "Brak wyników" : "No Analysis Loaded"}</p>
                    <p style={{ margin: "4px 0 0", fontSize: 14 }}>
                      {uiLang === "pl" ? "Wpisz tekst i kliknij 'Analizuj komentarz', aby zobaczyć wykresy." : "Input or select a comment and click 'Analyze Toxicity' to visualize probabilities."}
                    </p>
                  </div>
                )}

                {loading && (
                  <div style={{ textAlign: "center", padding: "64px 16px", color: "#64748b" }}>
                    <div style={{ fontSize: 32, marginBottom: 12, animation: "spin 1s linear infinite" }} className="animate-spin">
                      ⏳
                    </div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 16 }}>{uiLang === "pl" ? "Przetwarzanie..." : "Processing text..."}</p>
                    <p style={{ margin: "4px 0 0", fontSize: 14 }}>
                      {uiLang === "pl" ? "Uruchamianie wnioskowania " : "Running "}{MODEL_LABELS[model]}{uiLang === "pl" ? "" : " inference"}
                    </p>
                  </div>
                )}

                {result && (
                  <div style={{ position: "relative" }}>
                    <div
                      style={{
                        padding: "8px 12px",
                        backgroundColor: commentIsSafe ? "#f0fdf4" : "#fef2f2",
                        borderRadius: 8,
                        marginBottom: 16,
                        borderLeft: `4px solid ${commentIsSafe ? "#10b981" : "#ef4444"}`,
                        fontSize: 13,
                        color: "#475569",
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 8,
                        justifyContent: "space-between",
                      }}
                    >
                      <span>
                        {uiLang === "pl" ? "Werdykt:" : "Verdict:"}{" "}
                        <strong style={{ color: commentIsSafe ? "#059669" : "#dc2626" }}>
                          {commentIsSafe
                            ? uiLang === "pl"
                              ? "🟢 Bezpieczny komentarz"
                              : "🟢 Safe comment"
                            : uiLang === "pl"
                              ? "🔴 Wykryto naruszenie"
                              : "🔴 Toxic content detected"}
                        </strong>
                      </span>
                      <span>{uiLang === "pl" ? "Tryb modelu:" : "Model Mode:"} <strong>{MODEL_LABELS[result.model]}</strong></span>
                      {result.requested_lang === "auto" && result.analysis_lang && (
                        <span>
                          {uiLang === "pl" ? "Wykryty język:" : "Detected language:"}{" "}
                          <strong>
                            {result.analysis_lang === "pl" ? "Polski" : "English"}
                            {result.lang_confidence != null ? ` (${(result.lang_confidence * 100).toFixed(0)}%)` : ""}
                          </strong>
                          {result.lang_source && result.lang_source !== "forced" && (
                            <span style={{ fontSize: 11, color: "#94a3b8", marginLeft: 4 }}>({result.lang_source})</span>
                          )}
                        </span>
                      )}
                      <span>{uiLang === "pl" ? "Maks. wynik:" : "Max score:"} <strong>{(Math.max(...Object.values(result.probabilities)) * 100).toFixed(1)}%</strong></span>
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
                          <span style={{ fontSize: 12, fontWeight: "bold", color: "#8b5cf6" }}>{effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT"}</span>
                        </div>
                      </div>
                    )}

                    {/* Interactive Radar Chart Container */}
                    <div style={{ display: "flex", justifyContent: "center", marginBottom: 24, position: "relative" }}>
                      <svg width="300" height="300" style={{ overflow: "visible" }}>
                        {/* Background Polygons for Scale levels */}
                        {[0.2, 0.4, 0.6, 0.8, 1.0].map((level) => {
                          const levelPoints = Array.from({ length: numVertices })
                            .map((_, j) => {
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
                        {Array.from({ length: numVertices }).map((_, idx) => {
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
                          // Determine alignment based on position
                          const angleRad = (idx * Math.PI * 2) / numVertices - Math.PI / 2;
                          const cosA = Math.cos(angleRad);
                          const textAnchor = Math.abs(cosA) < 0.1 ? "middle" : cosA > 0 ? "start" : "end";

                          const labelVariant = result.is_dual
                            ? getLabelRowVariant(
                                label,
                                effectiveAnalysisLang,
                                result.probabilities_bert?.[label] ?? 0,
                                result.probabilities_tfidf?.[label] ?? 0,
                              )
                            : getLabelRowVariant(label, effectiveAnalysisLang, result.probabilities[label] ?? 0);
                          const isHighlighted = labelVariant !== "neutral";

                          return (
                            <text
                              key={label}
                              x={textCoords.x}
                              y={textCoords.y + 4}
                              textAnchor={textAnchor}
                              fill={
                                labelVariant === "safe"
                                  ? "#10b981"
                                  : labelVariant === "violation"
                                    ? "#ef4444"
                                    : "#475569"
                              }
                              fontSize="11"
                              fontWeight={isHighlighted ? "bold" : "600"}
                              style={{ transition: "all 0.2s ease" }}
                            >
                              {label.replace("_", " ")}
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
                              fill={commentIsSafe ? "#86efac77" : "#fca5a577"}
                              stroke={commentIsSafe ? "#10b981" : "#ef4444"}
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
                            const variant = getLabelRowVariant(pt.label, effectiveAnalysisLang, pt.value);
                            const dotFill =
                              variant === "safe" ? "#10b981" : variant === "violation" ? "#ef4444" : "#0284c7";
                            return (
                              <circle
                                key={pt.label}
                                cx={pt.x}
                                cy={pt.y}
                                r={isHovered ? 8 : 5}
                                fill={dotFill}
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
                            <span style={{ fontWeight: 800, textTransform: "uppercase" }}>{predictionTooltip.label.replace("_", " ")}</span>
                          </div>

                          {predictionTooltip.valueTfidf !== undefined && predictionTooltip.valueBert !== undefined ? (
                            // Dual Comparison Tooltip Info
                            <div style={{ display: "grid", gap: 4, marginBottom: 6 }}>
                              <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "#60a5fa", fontWeight: 600 }}>🔵 TF-IDF + LR:</span>
                                <strong>
                                  {(predictionTooltip.valueTfidf * 100).toFixed(1)}%
                                </strong>
                              </div>
                              <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "#c084fc", fontWeight: 600 }}>🟣 {effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT"}:</span>
                                <strong>
                                  {(predictionTooltip.valueBert * 100).toFixed(1)}%
                                </strong>
                              </div>
                            </div>
                          ) : (
                            // Single Model Tooltip Info
                            (() => {
                              const tooltipVariant = getLabelRowVariant(
                                predictionTooltip.label,
                                effectiveAnalysisLang,
                                predictionTooltip.value,
                              );
                              const isViolation = tooltipVariant === "violation";
                              const isSafe = tooltipVariant === "safe";
                              return (
                                <>
                                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                    <span>{uiLang === "pl" ? "Prawdopodobieństwo:" : "Score probability:"}</span>
                                    <strong
                                      style={{
                                        color: isViolation ? "#fca5a5" : isSafe ? "#86efac" : "#38bdf8",
                                      }}
                                    >
                                      {(predictionTooltip.value * 100).toFixed(1)}%
                                    </strong>
                                  </div>
                                  <div style={{ marginBottom: 6 }}>
                                    <strong>Status: </strong>
                                    <span
                                      style={{
                                        color: isViolation ? "#ef4444" : isSafe ? "#10b981" : "#94a3b8",
                                        fontWeight: "bold",
                                      }}
                                    >
                                      {isViolation
                                        ? uiLang === "pl"
                                          ? "🔴 NARUSZENIE"
                                          : "🔴 TOXIC"
                                        : isSafe
                                          ? uiLang === "pl"
                                            ? "🟢 BEZPIECZNY"
                                            : "🟢 SAFE"
                                          : uiLang === "pl"
                                            ? "⚪ NISKI"
                                            : "⚪ LOW"}
                                    </span>
                                  </div>
                                </>
                              );
                            })()
                          )}
                          <p style={{ margin: 0, color: "#cbd5e1", fontSize: 11 }}>
                            {LABEL_DESCRIPTIONS[predictionTooltip.label] ?? ""}
                          </p>
                        </div>
                      )}
                    </div>

                    <div style={{ textAlign: "center", fontSize: 12, color: "#64748b", margin: "-12px 0 20px" }}>
                      💡 <em>{uiLang === "pl" ? "Najedź na punkty na wykresie, aby porównać wyniki modeli." : "Hover over vertex points on the radar web to inspect side-by-side model predictions."}</em>
                    </div>

                    {/* Scores list */}
                    <div style={{ display: "grid", gap: 10 }}>
                      <label style={styles.label}>{uiLang === "pl" ? "Szczegółowa ocena klasyfikacji" : "Scores Breakdown & Model Contrast"}</label>
                      {sortedResult.map(([label]) => {
                        const labelIdx = radarLabels.indexOf(label);
                        const isHovered = predictionHoveredIdx === labelIdx;

                        if (result.is_dual && result.probabilities_tfidf && result.probabilities_bert) {
                          const pTfidf = result.probabilities_tfidf[label] ?? 0;
                          const pBert = result.probabilities_bert[label] ?? 0;
                          const rowVariant = getLabelRowVariant(label, effectiveAnalysisLang, pBert, pTfidf);

                          return (
                            <div
                              key={label}
                              style={{
                                ...styles.probabilityRow(rowVariant),
                                boxShadow: isHovered ? "0 0 8px rgba(139, 92, 246, 0.2)" : "none",
                                borderColor:
                                  isHovered
                                    ? "#8b5cf6"
                                    : rowVariant === "violation"
                                      ? "#fee2e2"
                                      : rowVariant === "safe"
                                        ? "#bbf7d0"
                                        : "#f1f5f9",
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
                                      backgroundColor: getBarColor(label, pTfidf, effectiveAnalysisLang, "#3b82f6"),
                                    }}
                                  />
                                </div>
                                <span style={{ fontSize: 11, fontWeight: "bold", textAlign: "right" }}>{(pTfidf * 100).toFixed(1)}%</span>
                              </div>

                              {/* Progress bar for BERT */}
                              <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 40px", gap: 8, alignItems: "center" }}>
                                <span style={{ fontSize: 11, color: "#475569" }}>🟣 {effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT"}</span>
                                <div style={{ height: 6, background: "#e2e8f0", borderRadius: 999, overflow: "hidden" }}>
                                  <div
                                    style={{
                                      height: "100%",
                                      width: `${pBert * 100}%`,
                                      borderRadius: 999,
                                      backgroundColor: getBarColor(label, pBert, effectiveAnalysisLang, "#8b5cf6"),
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
                          const rowVariant = getLabelRowVariant(label, effectiveAnalysisLang, p);
                          const isViolation = rowVariant === "violation";
                          const isSafe = rowVariant === "safe";
                          return (
                            <div
                              key={label}
                              style={{
                                ...styles.probabilityRow(rowVariant),
                                boxShadow: isHovered ? "0 0 8px rgba(59, 130, 246, 0.2)" : "none",
                                borderColor:
                                  isHovered
                                    ? "#3b82f6"
                                    : isViolation
                                      ? "#fee2e2"
                                      : isSafe
                                        ? "#bbf7d0"
                                        : "#f1f5f9",
                              }}
                              onMouseEnter={() => setPredictionHoveredIdx(labelIdx)}
                              onMouseLeave={() => setPredictionHoveredIdx(null)}
                            >
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <span style={{ fontWeight: 700, fontSize: 15, textTransform: "capitalize" }}>
                                  {label.replace("_", " ")}
                                </span>
                                <span
                                  style={{
                                    fontWeight: "bold",
                                    color: isViolation ? "#ef4444" : isSafe ? "#10b981" : "#475569",
                                    fontSize: 15,
                                  }}
                                >
                                  {(p * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div style={{ height: 8, background: "#e2e8f0", borderRadius: 999, overflow: "hidden" }}>
                                <div
                                  style={{
                                    height: "100%",
                                    width: `${p * 100}%`,
                                    borderRadius: 999,
                                    background: isViolation
                                      ? "linear-gradient(90deg, #f87171, #ef4444)"
                                      : isSafe
                                        ? "linear-gradient(90deg, #86efac, #10b981)"
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

          {/* SECTION: Semantic space maps (PCA validation or reference anchors) */}
          {result && (
            <div style={{ ...styles.card, marginTop: 24 }}>
              <div style={{ borderBottom: "1px solid #e2e8f0", paddingBottom: 12, marginBottom: 20 }}>
                <h3 style={{ margin: 0, fontSize: 20, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                  {mapVisualizationMode === "pca" ? "📊" : "🌐"}{" "}
                  {mapVisualizationMode === "pca"
                    ? (uiLang === "pl" ? "PCA — zbiór walidacyjny" : "PCA Validation Map")
                    : (uiLang === "pl" ? "Mapa referencyjna (kotwice)" : "Reference Anchor Map")}
                </h3>
                <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: 14 }}>
                  {mapVisualizationMode === "pca"
                    ? (uiLang === "pl"
                      ? "Prawdziwy scatter PCA/3D ze zbioru testowego. Kolor = poprawna klasyfikacja lub błąd (FP/FN)."
                      : "Real PCA scatter from the hold-out test set. Colors mark correct vs misclassified validation comments.")
                    : (uiLang === "pl"
                      ? "Klasyczna mapa z ustalonymi przykładami (safe / threat / hate). Pozycja Twojego tekstu z profilu prawdopodobieństw modelu."
                      : "Classic map with fixed benchmark comments. Your text is placed from the model's probability profile.")}
                </p>
                {mapVisualizationMode === "pca" && result.projection_method && (
                  <p style={{ margin: "6px 0 0", color: "#475569", fontSize: 12 }}>
                    {uiLang === "pl" ? "Metoda" : "Method"}: <strong>{result.projection_method}</strong>
                    {result.explained_variance_ratio && result.explained_variance_ratio.length >= 2 && (
                      <> · PC1+PC2 = {(100 * (result.explained_variance_ratio[0] + result.explained_variance_ratio[1])).toFixed(1)}% {uiLang === "pl" ? "wariancji" : "variance"}</>
                    )}
                  </p>
                )}
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16, alignItems: "center" }}>
                <div style={{ display: "flex", gap: 4, backgroundColor: "#e0e7ff", padding: 2, borderRadius: 6 }}>
                  <button
                    type="button"
                    disabled={!result.pca_included}
                    title={
                      result.pca_included
                        ? undefined
                        : uiLang === "pl"
                          ? "Włącz mapę PCA przy analizie (checkbox nad przyciskiem)"
                          : "Enable PCA map when analyzing (checkbox above Analyze button)"
                    }
                    style={{
                      padding: "4px 12px",
                      fontSize: 11,
                      fontWeight: "bold",
                      border: "none",
                      borderRadius: 4,
                      cursor: result.pca_included ? "pointer" : "not-allowed",
                      opacity: result.pca_included ? 1 : 0.45,
                      backgroundColor: mapVisualizationMode === "pca" ? "#4338ca" : "transparent",
                      color: mapVisualizationMode === "pca" ? "#fff" : "#475569",
                    }}
                    onClick={() => result.pca_included && setMapVisualizationMode("pca")}
                  >
                    PCA {uiLang === "pl" ? "walidacja" : "validation"}
                  </button>
                  <button
                    type="button"
                    style={{
                      padding: "4px 12px",
                      fontSize: 11,
                      fontWeight: "bold",
                      border: "none",
                      borderRadius: 4,
                      cursor: "pointer",
                      backgroundColor: mapVisualizationMode === "anchors" ? "#4338ca" : "transparent",
                      color: mapVisualizationMode === "anchors" ? "#fff" : "#475569",
                    }}
                    onClick={() => {
                      setMapVisualizationMode("anchors");
                      setProjectionDim(2);
                    }}
                  >
                    {uiLang === "pl" ? "Kotwice referencyjne" : "Reference anchors"}
                  </button>
                </div>
                {mapVisualizationMode === "pca" && (
                  <div style={{ display: "flex", gap: 4, backgroundColor: "#f1f5f9", padding: 2, borderRadius: 6 }}>
                    {([2, 3] as const).map((d) => (
                      <button
                        key={d}
                        type="button"
                        style={{
                          padding: "4px 10px",
                          fontSize: 11,
                          fontWeight: "bold",
                          border: "none",
                          borderRadius: 4,
                          cursor: "pointer",
                          backgroundColor: projectionDim === d ? "#0f172a" : "transparent",
                          color: projectionDim === d ? "#fff" : "#475569",
                        }}
                        onClick={() => setProjectionDim(d)}
                      >
                        {d === 3 ? (uiLang === "pl" ? "3D (obracaj)" : "3D (rotate)") : "2D"}
                      </button>
                    ))}
                  </div>
                )}
                {mapVisualizationMode === "pca" && (
                  <select
                    value={errorFilter}
                    onChange={(e) => setErrorFilter(e.target.value as ProjectionErrorFilter)}
                    style={{ fontSize: 12, padding: "4px 8px", borderRadius: 6, border: "1px solid #cbd5e1" }}
                  >
                    <option value="all">{uiLang === "pl" ? "Wszystkie punkty" : "All validation points"}</option>
                    <option value="correct">{uiLang === "pl" ? "Tylko poprawne" : "Correct only"}</option>
                    <option value="errors">{uiLang === "pl" ? "Tylko błędy" : "Misclassifications only"}</option>
                    <option value="false_positive">False Positive</option>
                    <option value="false_negative">False Negative</option>
                    <option value="label_mismatch">{uiLang === "pl" ? "Zła etykieta" : "Label mismatch"}</option>
                  </select>
                )}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
                {/* PCA scatter plot */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
                  <span style={styles.label}>
                    {mapVisualizationMode === "anchors"
                      ? (uiLang === "pl" ? "Mapa 2D z kotwicami" : "2D anchor map")
                      : projectionDim === 3
                        ? (uiLang === "pl" ? "Interaktywny wykres 3D PCA" : "Interactive 3D PCA plot")
                        : (uiLang === "pl" ? "Wykres 2D PCA" : "2D PCA scatter")}
                  </span>

                  {mapVisualizationMode === "pca" && projectionDim === 3 ? (
                    <div style={{ width: "100%", maxWidth: 520, border: "1px solid #cbd5e1", borderRadius: 12, overflow: "hidden" }}>
                      <ProjectionPlot3D
                        points={displayedProjectionPoints}
                        axisLabels={{ x: axisLabelX, y: axisLabelY, z: axisLabelZ }}
                        getColor={plot3DColor}
                        uiLang={uiLang}
                        height={420}
                      />
                    </div>
                  ) : (
                  <div style={{ position: "relative", width: 320, height: 320, border: "1px solid #cbd5e1", borderRadius: 12, backgroundColor: "#f8fafc", overflow: "visible" }}>
                    <svg width="320" height="320" style={{ overflow: "visible" }}>
                      <line x1="160" y1="0" x2="160" y2="320" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="0" y1="160" x2="320" y2="160" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="3 3" />
                      {mapVisualizationMode === "anchors" ? (
                        <>
                          <text x="20" y="30" fill="#94a3b8" fontSize="9" fontWeight="bold">{uiLang === "pl" ? "🛡️ Bezpieczne" : "🛡️ Clean"}</text>
                          <text x="300" y="30" fill="#94a3b8" fontSize="9" fontWeight="bold" textAnchor="end">{uiLang === "pl" ? "⚠️ Przemoc/Hejt" : "⚠️ Threats & Hate"}</text>
                          <text x="300" y="300" fill="#94a3b8" fontSize="9" fontWeight="bold" textAnchor="end">{uiLang === "pl" ? "🤬 Wulgaryzmy" : "🤬 Insults & Obscene"}</text>
                        </>
                      ) : (
                        <>
                          <text x="8" y="312" fill="#64748b" fontSize="9" fontWeight="bold">{axisLabelX}</text>
                          <text x="8" y="14" fill="#64748b" fontSize="9" fontWeight="bold" transform="rotate(-90 8 14)">{axisLabelY}</text>
                        </>
                      )}

                      {result.is_dual && (() => {
                        const tfList = mapVisualizationMode === "anchors" ? result.reference_projection_tfidf : result.similarity_projection_tfidf;
                        const bertList = mapVisualizationMode === "anchors" ? result.reference_projection_bert : result.similarity_projection_bert;
                        const tfActive = tfList?.find(p => p.is_active);
                        const bertActive = bertList?.find(p => p.is_active);
                        if (tfActive && bertActive) {
                          const tfD = getDisplayCoords(tfActive);
                          const bertD = getDisplayCoords(bertActive);
                          const coordTf = getProjectionSVGCoords(tfD.x, tfD.y);
                          const coordBert = getProjectionSVGCoords(bertD.x, bertD.y);
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

                      {displayedProjectionPoints.map((pt) => {
                        if (pt.is_active) return null;
                        const disp = getDisplayCoords(pt);
                        const { cx, cy } = getProjectionSVGCoords(disp.x, disp.y);
                        const isHovered = hoveredProjectionId === pt.id;
                        const color = getPointColor(pt);

                        return (
                          <circle
                            key={pt.id}
                            cx={cx}
                            cy={cy}
                            r={isHovered ? (mapVisualizationMode === "anchors" ? 9 : 6) : (mapVisualizationMode === "anchors" ? 5.5 : 3.5)}
                            fill={color}
                            stroke="#fff"
                            strokeWidth={isHovered ? 1.5 : 0.75}
                            opacity={hoveredProjectionId && !isHovered ? 0.25 : 0.7}
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
                            const tfActive = (mapVisualizationMode === "anchors" ? result.reference_projection_tfidf : result.similarity_projection_tfidf)?.find(p => p.is_active);
                            if (!tfActive) return null;
                            const tfD = getDisplayCoords(tfActive);
                            const { cx, cy } = getProjectionSVGCoords(tfD.x, tfD.y);
                            const tfColor = getActiveUserColor(tfActive);
                            return (
                              <g key="active_user_tfidf">
                                <circle cx={cx} cy={cy} r="12" fill="none" stroke={tfColor} strokeWidth="1.5" opacity="0.5">
                                  <animate attributeName="r" values="6;14;6" dur="2.5s" repeatCount="indefinite" />
                                </circle>
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r="7"
                                  fill={tfColor}
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
                            const bertActive = (mapVisualizationMode === "anchors" ? result.reference_projection_bert : result.similarity_projection_bert)?.find(p => p.is_active);
                            if (!bertActive) return null;
                            const bertD = getDisplayCoords(bertActive);
                            const { cx, cy } = getProjectionSVGCoords(bertD.x, bertD.y);
                            const bertColor = getActiveUserColor(bertActive);
                            return (
                              <g key="active_user_bert">
                                <circle cx={cx} cy={cy} r="14" fill="none" stroke={bertColor} strokeWidth="1.5" opacity="0.6">
                                  <animate attributeName="r" values="7;16;7" dur="2s" repeatCount="indefinite" />
                                </circle>
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r="8"
                                  fill={bertColor}
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
                          const activePt = (mapVisualizationMode === "anchors" ? result.reference_projection : result.similarity_projection)?.find(p => p.is_active);
                          if (!activePt) return null;
                          const activeD = getDisplayCoords(activePt);
                          const { cx, cy } = getProjectionSVGCoords(activeD.x, activeD.y);
                          const activeColor = getActiveUserColor(activePt);
                          return (
                            <g key="active_user_single">
                              <circle cx={cx} cy={cy} r="16" fill="none" stroke={activeColor} strokeWidth="2" opacity="0.4">
                                <animate attributeName="r" values="8;18;8" dur="2s" repeatCount="indefinite" />
                                <animate attributeName="opacity" values="0.7;0.1;0.7" dur="2s" repeatCount="indefinite" />
                              </circle>
                              <circle
                                cx={cx}
                                cy={cy}
                                r="8"
                                fill={activeColor}
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
                              ? (projectionTooltip.modelType === "tfidf" ? "🔵 YOUR TEXT (TF-IDF)" : projectionTooltip.modelType === "bert" ? "🟣 YOUR TEXT (BERT)" : "⭐ YOUR TEXT")
                              : projectionTooltip.point.is_validation
                                ? `📍 VALIDATION · ${projectionTooltip.point.error_type ?? "—"}`
                                : "📌 POINT"}
                          </span>
                          {!projectionTooltip.point.is_active && projectionTooltip.point.similarity > 0 && (
                            <span style={{ fontWeight: 800, color: "#60a5fa" }}>
                              sim: {(projectionTooltip.point.similarity * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        <p style={{ margin: "0 0 6px", fontStyle: "italic", color: "#e2e8f0", fontSize: 11 }}>
                          "{projectionTooltip.point.text.length > 90 ? `${projectionTooltip.point.text.substring(0, 87)}...` : projectionTooltip.point.text}"
                        </p>
                        {projectionTooltip.point.is_validation && (
                          <div style={{ fontSize: 10, color: "#cbd5e1", marginBottom: 6 }}>
                            <div>GT: {(projectionTooltip.point.ground_truth_labels ?? []).join(", ") || "safe"}</div>
                            <div>Pred: {(projectionTooltip.point.predicted_labels ?? projectionTooltip.point.labels).join(", ")}</div>
                          </div>
                        )}
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
                  )}

                  {/* Legend */}
                  <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10, marginTop: 12, fontSize: 11, fontWeight: 600 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: commentIsSafe ? "#10b981" : "#ef4444", border: "2px solid #fff", boxShadow: "0 0 0 1px #cbd5e1" }} />
                      <span>{uiLang === "pl" ? "Twój komentarz" : "Your comment"}</span>
                    </div>
                    {mapVisualizationMode === "pca" ? (
                      <>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#94a3b8" }} />
                          <span>{uiLang === "pl" ? "Poprawne" : "Correct"}</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#ef4444" }} />
                          <span>FP</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#f97316" }} />
                          <span>FN</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#a855f7" }} />
                          <span>{uiLang === "pl" ? "Zła etykieta" : "Label err."}</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#10b981" }} />
                          <span>{uiLang === "pl" ? "Bezpieczne" : "Clean"}</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#f97316" }} />
                          <span>{uiLang === "pl" ? "Przemoc" : "Threat"}</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#ec4899" }} />
                          <span>{uiLang === "pl" ? "Wulgaryzmy" : "Obscene"}</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Closest Matches Card List Column */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <span style={styles.label}>
                      {mapVisualizationMode === "anchors"
                        ? (uiLang === "pl" ? "Najbardziej podobne wzorce" : "Closest Reference Comments")
                        : (uiLang === "pl" ? "Najbliższe ze zbioru walidacyjnego" : "Nearest Validation Comments")}
                    </span>
                    
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
                          {effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT"}
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
                          title={uiLang === "pl" ? "Kliknij, aby załadować ten tekst do analizatora!" : "Click to load this comment into the analyzer!"}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: closestMatchesModel === "bert" && result.is_dual ? "#7c3aed" : "#2563eb" }}>
                              🔥 {uiLang === "pl" ? "Podobieństwo:" : "Similarity Match:"} {(match.similarity * 100).toFixed(1)}%
                            </span>
                            <span style={{ fontSize: 11, color: "#64748b", textDecoration: "underline" }}>
                              {uiLang === "pl" ? "Przetestuj 🚀" : "Click to test 🚀"}
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
                    💡 <em>{uiLang === "pl" ? (effectiveAnalysisLang === "pl" ? "Kliknięcie w kartę wzorca automatycznie skopiuje polski tekst do analizatora." : "Kliknięcie w kartę wzorca skopiuje angielski tekst do analizatora.") : "Clicking any reference card above copies it to the analyzer, helping you observe how different models respond to it."}</em>
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
              ⏳ {uiLang === "pl" ? "Wczytywanie statystyk..." : "Loading model metrics..."}
            </div>
          )}

          {!loadingMetrics && !activeMetrics && (
            <div style={{ textAlign: "center", padding: "48px 16px", color: "#ef4444" }}>
              <p style={{ margin: "0 0 12px" }}>
                ⚠️{" "}
                {metricsError ??
                  (uiLang === "pl"
                    ? "Nie udało się załadować statystyk z serwera."
                    : "Failed to load model metrics from backend.")}
              </p>
              <button
                type="button"
                onClick={() => void loadMetrics(true)}
                style={{
                  padding: "8px 16px",
                  borderRadius: 8,
                  border: "none",
                  backgroundColor: "#2563eb",
                  color: "#fff",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {uiLang === "pl" ? "🔄 Spróbuj ponownie" : "🔄 Retry"}
              </button>
            </div>
          )}

          {activeMetrics && (
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
                    🔵 TF-IDF + Logistic Regression ({effectiveAnalysisLang.toUpperCase()})
                  </h3>
                  <p style={{ fontSize: 14, color: "#64748b", margin: "0 0 16px" }}>
                    {uiLang === "pl" 
                      ? "Klasyczny, bardzo szybki model oparty na częstości słów i n-gramów znakowych." 
                      : "Baseline classical machine learning pipeline. Super fast training and inference, low resource consumption."}
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MACRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#1e3a8a" }}>
                        {(activeMetrics.tfidf_lr.f1_macro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MICRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#1e3a8a" }}>
                        {(activeMetrics.tfidf_lr.f1_micro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>HAMMING LOSS</span>
                      <strong style={{ fontSize: 20, color: "#1e3a8a" }}>
                        {activeMetrics.tfidf_lr.hamming_loss.toFixed(4)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>TEST SAMPLES</span>
                      <strong style={{ fontSize: 20, color: "#1e3a8a" }}>
                        {activeMetrics.tfidf_lr.dataset.n_test.toLocaleString()}
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
                    🟣 {effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT"} (Transformer) Model ({effectiveAnalysisLang.toUpperCase()})
                  </h3>
                  <p style={{ fontSize: 14, color: "#64748b", margin: "0 0 16px" }}>
                    {uiLang === "pl"
                      ? "Zaawansowana sieć głęboka fine-tunowana na polskim korpusie BAN-PL dla pełnego zrozumienia kontekstu."
                      : "Fine-tuned context-aware encoder. Slower inference but superior language understanding and higher accuracy."}
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MACRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#4c1d95" }}>
                        {(activeMetrics.bert.f1_macro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>MICRO F1-SCORE</span>
                      <strong style={{ fontSize: 24, color: "#4c1d95" }}>
                        {(activeMetrics.bert.f1_micro * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>HAMMING LOSS</span>
                      <strong style={{ fontSize: 20, color: "#4c1d95" }}>
                        {activeMetrics.bert.hamming_loss.toFixed(4)}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: 12, color: "#64748b", display: "block" }}>TEST SAMPLES</span>
                      <strong style={{ fontSize: 20, color: "#4c1d95" }}>
                        {activeMetrics.bert.dataset.n_test.toLocaleString()}
                      </strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* Grouped Bar Chart of Class Metrics */}
              <div style={styles.card}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 20 }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>{uiLang === "pl" ? "Porównanie klasyfikacji według kategorii" : "Class-by-Class Metric Comparison"}</h3>
                    <p style={{ margin: "2px 0 0", color: "#64748b", fontSize: 13 }}>
                      {uiLang === "pl" ? "Porównaj wyniki Precision, Recall oraz F1-Score dla każdej kategorii wypowiedzi." : "Compare precision, recall, and f1-score across classes."}
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
                    {activeMetrics.tfidf_lr.per_label.map((tfidfInfo, idx) => {
                      const label = tfidfInfo.label;
                      const bertInfo = activeMetrics.bert.per_label.find((l) => l.label === label) || tfidfInfo;

                      // Extract value
                      const tfidfVal = tfidfInfo[selectedMetric];
                      const bertVal = bertInfo[selectedMetric];

                      // Dynamic Spacing and sizing of columns based on how many categories we have
                      const barCount = activeMetrics.tfidf_lr.per_label.length;
                      const groupWidth = barCount === 4 ? 130 : 90;
                      const offsetPadding = barCount === 4 ? 40 : 15;
                      const groupX = 60 + idx * groupWidth + offsetPadding;

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
                        <span style={{ fontWeight: 800, textTransform: "uppercase" }}>{metricsHoveredBar.label.replace("_", " ")}</span>
                        <span style={{ fontWeight: 800, color: metricsHoveredBar.modelId === "bert" ? "#c084fc" : "#60a5fa" }}>
                          {(metricsHoveredBar.value * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ marginBottom: 4 }}>
                        <strong>Model: </strong>
                        {metricsHoveredBar.modelId === "bert" ? (effectiveAnalysisLang === "pl" ? "HerBERT" : "BERT") : "TF-IDF + LR"}
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
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#475569" }}>{effectiveAnalysisLang === "pl" ? "HerBERT (Fine-tuned Transformer)" : "BERT (Fine-tuned Transformer)"}</span>
                  </div>
                </div>
              </div>

              {/* Analysis and Insights */}
              <div style={styles.card}>
                <h3 style={{ margin: "0 0 12px", fontSize: 18, fontWeight: 700 }}>🔍 {uiLang === "pl" ? "Kluczowe spostrzeżenia" : "Key Evaluation Insights"}</h3>
                {uiLang === "pl" ? (
                  effectiveAnalysisLang === "pl" ? (
                    <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10, fontSize: 14, color: "#475569" }}>
                      <li>
                        <strong>Model TF-IDF:</strong> Na polskim korpusie BAN-PL, TF-IDF osiąga silne i zbalansowane wyniki (Macro F1: <strong>{(activeMetrics.tfidf_lr.f1_macro * 100).toFixed(1)}%</strong>), wykazując wyjątkowo wysoką precyzję dla klasy <code>safe</code> (<strong>85.9%</strong>) oraz zadowalające wykrywanie dla trudniejszych klas agresywnych.
                      </li>
                      <li>
                        <strong>Zbalansowany zbiór BAN-PL:</strong> Polski zbiór BAN-PL zawiera łącznie <strong>23 985 komentarzy</strong> z serwisu Wykop.pl, podzielonych dokładnie na wypowiedzi bezpieczne i szkodliwe. Dzięki temu modele uczą się stabilnego rozróżniania kontekstu bez faworyzowania jednej z klas.
                      </li>
                      <li>
                        <strong>Porównanie klasyfikacji:</strong> Kategorie <code>safe</code> (2397 próbek testowych) oraz <code>hate speech</code> (1350 próbek testowych) osiągają najwyższe wyniki F1 z racji licznej reprezentacji w korpusie treningowym. Rzadsze klasy, takie jak <code>violence</code> czy <code>vulgarity</code>, stanowią wyzwanie, w którym modele oparte o n-gramy znakowe sprawdzają się najlepiej dzięki wyłapywaniu ukrytych wulgaryzmów.
                      </li>
                    </ul>
                  ) : (
                    <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10, fontSize: 14, color: "#475569" }}>
                      <li>
                        <strong>Przewaga modelu Transformer:</strong> BERT znacząco przewyższa klasyczny model bazowy TF-IDF w prawie wszystkich klasach, osiągając wynik Macro F1-score na poziomie <strong>{(activeMetrics.bert.f1_macro * 100).toFixed(1)}%</strong> w porównaniu do <strong>{(activeMetrics.tfidf_lr.f1_macro * 100).toFixed(1)}%</strong> dla TF-IDF.
                      </li>
                      <li>
                        <strong>Wydajność w rzadkich klasach:</strong> Słabo reprezentowane kategorie, takie jak <code>threat</code>, odnotowują największy względny wzrost wydajności przy użyciu BERT dzięki głębokiemu zrozumieniu kontekstu semantycznego.
                      </li>
                      <li>
                        <strong>Kompromis Precision i Recall:</strong> Klasyczny model TF-IDF ma bardzo wysoki Recall dla klasy <code>toxic</code> z powodu zastosowania wag klasowych w celu redukcji False Negatives, lecz kosztem niższej precyzji. BERT osiąga znacznie zdrowszą równowagę, podnosząc precyzję przy jednoczesnym zachowaniu doskonałego poziomu czułości (Recall).
                      </li>
                    </ul>
                  )
                ) : (
                  effectiveAnalysisLang === "pl" ? (
                    <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10, fontSize: 14, color: "#475569" }}>
                      <li>
                        <strong>TF-IDF Model:</strong> On the Polish BAN-PL corpus, TF-IDF achieves strong and balanced results (Macro F1: <strong>{(activeMetrics.tfidf_lr.f1_macro * 100).toFixed(1)}%</strong>), showing exceptionally high precision for the <code>safe</code> class (<strong>85.9%</strong>) and satisfactory detection for aggressive classes.
                      </li>
                      <li>
                        <strong>Balanced BAN-PL Corpus:</strong> The Polish BAN-PL dataset contains a total of <strong>23,985 comments</strong> from Wykop.pl, split equally between safe and harmful comments. This helps the models learn clean boundary distinctions without bias.
                      </li>
                      <li>
                        <strong>Class Breakdown:</strong> The <code>safe</code> and <code>hate speech</code> classes see the highest F1 scores due to high representation in training data. Rarer classes like <code>violence</code> and <code>vulgarity</code> are more challenging, where character n-gram features prove highly effective at capturing toxic slurs.
                      </li>
                    </ul>
                  ) : (
                    <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 10, fontSize: 14, color: "#475569" }}>
                      <li>
                        <strong>Transformer Superiority:</strong> BERT outperforms the classical TF-IDF baseline significantly across almost all classes, achieving a Macro F1-score of <strong>{(activeMetrics.bert.f1_macro * 100).toFixed(1)}%</strong> compared to TF-IDF's <strong>{(activeMetrics.tfidf_lr.f1_macro * 100).toFixed(1)}%</strong>.
                      </li>
                      <li>
                        <strong>Rare Class Performance:</strong> Underrepresented categories like <code>threat</code> see the largest relative performance jumps with BERT due to contextual understanding.
                      </li>
                      <li>
                        <strong>Precision vs. Recall Trade-off:</strong> Classical TF-IDF has very high Recall on <code>toxic</code> because <code>class_weight="balanced"</code> bias was applied to reduce False Negatives, but at the cost of lower Precision. BERT achieves a much healthier balance, boosting Precision substantially while maintaining superb Recall.
                      </li>
                    </ul>
                  )
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

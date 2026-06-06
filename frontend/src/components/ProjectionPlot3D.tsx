import { useMemo } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export type PlotPoint = {
  id: string;
  text: string;
  labels: string[];
  x: number;
  y: number;
  z?: number;
  similarity: number;
  is_active: boolean;
  is_validation?: boolean;
  error_type?: string | null;
  ground_truth_labels?: string[];
  predicted_labels?: string[];
};

type Props = {
  points: PlotPoint[];
  axisLabels?: { x?: string; y?: string; z?: string };
  getColor: (pt: PlotPoint) => string;
  height?: number;
  uiLang?: "en" | "pl";
};

function pointHover(pt: PlotPoint, uiLang: "en" | "pl"): string {
  const labels = pt.labels.join(", ") || "safe";
  if (pt.is_active) {
    return uiLang === "pl" ? `<b>Twój tekst</b><br>${labels}` : `<b>Your text</b><br>${labels}`;
  }
  const err = pt.error_type ? `<br>error: ${pt.error_type}` : "";
  const gt = pt.ground_truth_labels?.length ? `<br>GT: ${pt.ground_truth_labels.join(", ")}` : "";
  const pred = pt.predicted_labels?.length ? `<br>Pred: ${pt.predicted_labels.join(", ")}` : "";
  const short = pt.text.length > 80 ? `${pt.text.slice(0, 77)}...` : pt.text;
  return `<b>${labels}</b>${err}${gt}${pred}<br><i>${short}</i>`;
}

export default function ProjectionPlot3D({
  points,
  axisLabels,
  getColor,
  height = 420,
  uiLang = "en",
}: Props) {
  const { validationTrace, activeTrace } = useMemo(() => {
    const validation = points.filter((p) => !p.is_active);
    const active = points.filter((p) => p.is_active);

    const validationTrace = {
      type: "scatter3d" as const,
      mode: "markers" as const,
      name: uiLang === "pl" ? "Zbiór walidacyjny" : "Validation set",
      x: validation.map((p) => p.x),
      y: validation.map((p) => p.y),
      z: validation.map((p) => p.z ?? 0),
      text: validation.map((p) => pointHover(p, uiLang)),
      hoverinfo: "text" as const,
      marker: {
        size: 3,
        color: validation.map((p) => getColor(p)),
        opacity: 0.75,
        line: { width: 0 },
      },
    };

    const activeTrace = {
      type: "scatter3d" as const,
      mode: "markers" as const,
      name: uiLang === "pl" ? "Twój komentarz" : "Your comment",
      x: active.map((p) => p.x),
      y: active.map((p) => p.y),
      z: active.map((p) => p.z ?? 0),
      text: active.map((p) => pointHover(p, uiLang)),
      hoverinfo: "text" as const,
      marker: {
        size: 9,
        color: active.map((p) => getColor(p)),
        opacity: 1,
        line: { color: "#ffffff", width: 2 },
        symbol: "diamond" as const,
      },
    };

    return { validationTrace, activeTrace };
  }, [points, getColor, uiLang]);

  return (
    <Plot
      data={[validationTrace, activeTrace]}
      layout={{
        autosize: true,
        height,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        paper_bgcolor: "#f8fafc",
        scene: {
          bgcolor: "#f8fafc",
          xaxis: { title: axisLabels?.x ?? "PC1", backgroundcolor: "#f1f5f9" },
          yaxis: { title: axisLabels?.y ?? "PC2", backgroundcolor: "#f1f5f9" },
          zaxis: { title: axisLabels?.z ?? "PC3", backgroundcolor: "#f1f5f9" },
          camera: { eye: { x: 1.6, y: 1.4, z: 1.2 } },
        },
        legend: { orientation: "h", y: 1.08 },
        title: {
          text: uiLang === "pl" ? "Obróć myszą · przybliż kółkiem" : "Drag to rotate · scroll to zoom",
          font: { size: 11, color: "#64748b" },
        },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
      }}
      style={{ width: "100%", minHeight: height }}
      useResizeHandler
    />
  );
}

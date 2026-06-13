import { useEffect, useState } from "react";
import { Activity, BarChart3, Database } from "lucide-react";
import { predictImageWithGradCam } from "./api/predictApi";
import DisclaimerBox from "./components/DisclaimerBox.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import GradCamViewer from "./components/GradCamViewer.jsx";

const MODEL_INFO = [
  { label: "Train", value: "15,000" },
  { label: "Valid", value: "202" },
  { label: "Labels", value: "14" },
  { label: "Model", value: "EffNet-B0" },
  { label: "Size", value: "320px" },
  { label: "Best AUROC", value: "0.836" },
  { label: "Mean AP", value: "0.523" },
  { label: "Best epoch", value: "4/5" },
];

const LABEL_INFO = [
  { label: "No Finding", train: "3,250", auc: "0.823" },
  { label: "Enlarged Cardio", train: "961", auc: "0.660" },
  { label: "Cardiomegaly", train: "2,230", auc: "0.791" },
  { label: "Lung Opacity", train: "7,072", auc: "0.899" },
  { label: "Lung Lesion", train: "770", auc: "0.697" },
  { label: "Edema", train: "4,194", auc: "0.895" },
  { label: "Consolidation", train: "1,464", auc: "0.867" },
  { label: "Pneumonia", train: "994", auc: "0.818" },
  { label: "Atelectasis", train: "3,032", auc: "0.856" },
  { label: "Pneumothorax", train: "1,584", auc: "0.826" },
  { label: "Pleural Eff.", train: "5,918", auc: "0.883" },
  { label: "Pleural Other", train: "413", auc: "0.955*" },
  { label: "Fracture", train: "710", auc: "N/A" },
  { label: "Support Dev.", train: "9,557", auc: "0.896" },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  async function runPrediction() {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictImageWithGradCam(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero-content">
          <div className="hero-badge">
            <Activity size={18} />
            Human-centered medical AI prototype
          </div>
          <h1>MediVision Quality Lab</h1>
          <p>
            A research demo for CheXpert multi-label chest X-ray classification,
            Grad-CAM explanation, and medical AI deployment infrastructure.
          </p>
        </div>

        <aside className="hero-stats" aria-label="Model training summary">
          <div className="hero-stats-topline">
            <div className="hero-stats-title">
              <BarChart3 size={18} />
              Model & label summary
            </div>
            <div className="hero-stats-note">
              <Database size={14} />
              CheXpert subset · frontal-only · uncertain→0
            </div>
          </div>

          <div className="hero-stats-grid">
            {MODEL_INFO.map((item) => (
              <div className="hero-stat-item" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>

          <div className="label-summary-header">
            <span>Per-label: <strong>Train positives / AUROC</strong></span>
            <small>* valid positives too few; interpret cautiously</small>
          </div>
          <div className="label-summary-grid">
            {LABEL_INFO.map((item) => (
              <div className="label-summary-item" key={item.label}>
                <span>{item.label}</span>
                <div className="label-summary-values">
                  <strong>{item.train}</strong>
                  <em>{item.auc}</em>
                </div>
              </div>
            ))}
          </div>
        </aside>
      </header>

      <DisclaimerBox />

      <div className="grid-layout">
        <ImageUploader
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onFileChange={setSelectedFile}
          onRun={runPrediction}
          loading={loading}
        />

        <PredictionCard result={result} />
        <GradCamViewer result={result} />
      </div>

      {error && (
        <section className="error-box">
          <strong>Prediction failed</strong>
          <pre>{error}</pre>
        </section>
      )}
    </main>
  );
}

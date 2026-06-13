import { useEffect, useState } from "react";
import { Activity, BarChart3, Database } from "lucide-react";
import { predictImageWithGradCam } from "./api/predictApi";
import DisclaimerBox from "./components/DisclaimerBox.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import GradCamViewer from "./components/GradCamViewer.jsx";

const MODEL_INFO = [
  { label: "Train images", value: "15,000" },
  { label: "Valid images", value: "202" },
  { label: "Labels", value: "14" },
  { label: "Model", value: "EfficientNet-B0" },
  { label: "Input size", value: "320×320" },
  { label: "Best AUROC", value: "0.836" },
  { label: "Mean AP", value: "0.523" },
  { label: "Best epoch", value: "4 / 5" },
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
          <div className="hero-stats-title">
            <BarChart3 size={18} />
            Model summary
          </div>
          <div className="hero-stats-grid">
            {MODEL_INFO.map((item) => (
              <div className="hero-stat-item" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className="hero-stats-note">
            <Database size={14} />
            CheXpert subset, frontal-only, uncertain labels mapped to zero.
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

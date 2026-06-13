import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, Server } from "lucide-react";
import { API_BASE_URL, predictImageWithGradCam } from "./api/predictApi";
import DisclaimerBox from "./components/DisclaimerBox.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import GradCamViewer from "./components/GradCamViewer.jsx";

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

  const modelReadyHint = useMemo(() => {
    return "Place models/pneumonia_resnet18_224.pt before running prediction.";
  }, []);

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
        <div className="hero-badge">
          <Activity size={18} />
          Human-centered medical AI prototype
        </div>
        <h1>MediVision Quality Lab</h1>
        <p>
          A research demo for MedMNIST image classification, Grad-CAM explanation,
          and Kubernetes-based AI deployment infrastructure.
        </p>
      </header>

      <DisclaimerBox />

      <div className="status-strip">
        <div>
          <Server size={18} />
          API: <code>{API_BASE_URL}</code>
        </div>
        <div>
          <AlertTriangle size={18} />
          {modelReadyHint}
        </div>
      </div>

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

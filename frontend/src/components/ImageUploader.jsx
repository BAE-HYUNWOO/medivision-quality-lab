import { UploadCloud } from "lucide-react";

export default function ImageUploader({ selectedFile, previewUrl, onFileChange, onRun, loading }) {
  return (
    <section className="card uploader-card">
      <div className="card-header">
        <UploadCloud size={22} />
        <div>
          <h2>Upload medical image</h2>
          <p>Use a MedMNIST-style image or any demo image for testing.</p>
        </div>
      </div>

      <label className="dropzone">
        <input
          type="file"
          accept="image/*"
          onChange={(event) => onFileChange(event.target.files?.[0] || null)}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="Selected preview" className="preview-image" />
        ) : (
          <div className="dropzone-placeholder">
            <span>Click to select an image</span>
            <small>PNG/JPG recommended</small>
          </div>
        )}
      </label>

      <button className="primary-button" disabled={!selectedFile || loading} onClick={onRun}>
        {loading ? "Running model..." : "Predict with Grad-CAM"}
      </button>
    </section>
  );
}

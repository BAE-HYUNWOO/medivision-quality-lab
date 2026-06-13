import { useState } from "react";
import { UploadCloud } from "lucide-react";

export default function ImageUploader({ selectedFile, previewUrl, onFileChange, onRun, loading }) {
  const [dragging, setDragging] = useState(false);

  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);

    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("이미지 파일만 업로드할 수 있습니다.");
      return;
    }

    onFileChange(file);
  }

  return (
    <section className="card uploader-card">
      <div className="card-header">
        <UploadCloud size={22} />
        <div>
          <h2>Upload medical image</h2>
          <p>Click to select, or drag and drop a chest X-ray image here.</p>
        </div>
      </div>

      <label
        className={`dropzone ${dragging ? "is-dragging" : ""}`}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          onChange={(event) => onFileChange(event.target.files?.[0] || null)}
        />
        {previewUrl ? (
          <div className="preview-frame">
            <img src={previewUrl} alt="Selected preview" className="preview-image" />
            <div className="dropzone-hint">Click or drag another image to replace</div>
          </div>
        ) : (
          <div className="dropzone-placeholder">
            <UploadCloud size={38} />
            <span>Drag X-ray image here</span>
            <small>or click to select PNG/JPG</small>
          </div>
        )}
      </label>

      <button className="primary-button" disabled={!selectedFile || loading} onClick={onRun}>
        {loading ? "Running model..." : "Predict with Grad-CAM"}
      </button>
    </section>
  );
}

import { useState, useCallback } from "react";
import { predictDisease } from "../utils/api";

export default function UploadPage({ onResult }) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setFile(f);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const onInputChange = (e) => handleFile(e.target.files[0]);

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const result = await predictDisease(file);
      onResult({ ...result, originalImage: preview });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page upload-page">
      <div className="upload-hero">
        <span className="upload-icon">🌿</span>
        <h1>CropSense AI</h1>
        <p className="subtitle">Upload a leaf photo — detect disease in seconds</p>
      </div>

      <div
        className={`drop-zone ${dragOver ? "drag-over" : ""} ${preview ? "has-preview" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById("file-input").click()}
      >
        {preview ? (
          <img src={preview} alt="Preview" className="preview-img" />
        ) : (
          <div className="drop-placeholder">
            <span className="drop-icon">📷</span>
            <p>Drag & drop a leaf image here</p>
            <p className="drop-hint">or click to browse — JPG, PNG, WEBP</p>
          </div>
        )}
        <input
          id="file-input"
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={onInputChange}
        />
      </div>

      {error && <div className="error-msg">⚠️ {error}</div>}

      {preview && (
        <div className="upload-actions">
          <button className="btn-secondary" onClick={() => { setPreview(null); setFile(null); }}>
            Clear
          </button>
          <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? (
              <span className="btn-loading"><span className="spinner" /> Analyzing...</span>
            ) : "Analyze Disease"}
          </button>
        </div>
      )}

      <div className="supported-plants">
        <p className="section-label">Supports 14 plants · 38 disease classes</p>
        <div className="plant-chips">
          {["🍅 Tomato", "🌽 Corn", "🍇 Grape", "🥔 Potato", "🍎 Apple",
            "🫑 Pepper", "🍓 Strawberry", "🌸 Peach", "🌿 Soybean"].map(p => (
            <span key={p} className="chip">{p}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ResultPage({ result, onBack }) {
  if (!result) return null;

  const {
    plant, disease, confidence, is_healthy,
    top_predictions, gradcam_heatmap, treatment,
    needs_expert_review, originalImage
  } = result;

  const severityColor = {
    none: "#22c55e",
    low: "#86efac",
    moderate: "#f59e0b",
    high: "#ef4444",
  };

  return (
    <div className="page result-page">
      <button className="back-btn" onClick={onBack}>← New Analysis</button>

      <div className="result-header">
        <div className={`status-badge ${is_healthy ? "healthy" : "diseased"}`}>
          {is_healthy ? "✅ Healthy" : "⚠️ Disease Detected"}
        </div>
        <h2>{plant}</h2>
        <p className="disease-name">{disease}</p>
        <div className="confidence-bar-wrap">
          <div className="confidence-bar" style={{ width: `${confidence}%` }} />
          <span className="confidence-label">{confidence}% confidence</span>
        </div>
        {needs_expert_review && (
          <div className="expert-alert">
            🔬 Low confidence — recommend expert review
          </div>
        )}
      </div>

      <div className="result-grid">
        {/* Image comparison */}
        <div className="card images-card">
          <h3>Visual Analysis</h3>
          <div className="image-compare">
            <div className="img-wrap">
              <img src={originalImage} alt="Original" />
              <span className="img-label">Original</span>
            </div>
            <div className="img-wrap">
              <img src={`data:image/png;base64,${gradcam_heatmap}`} alt="Grad-CAM" />
              <span className="img-label">Grad-CAM Heatmap</span>
            </div>
          </div>
          <p className="gradcam-note">
            🔴 Red regions = areas that most influenced the prediction
          </p>
        </div>

        {/* Top predictions */}
        <div className="card predictions-card">
          <h3>Top Predictions</h3>
          <div className="predictions-list">
            {top_predictions.map((p, i) => (
              <div key={i} className={`prediction-item ${i === 0 ? "top" : ""}`}>
                <span className="pred-rank">#{i + 1}</span>
                <div className="pred-info">
                  <span className="pred-name">{p.class.replace("___", " → ").replace(/_/g, " ")}</span>
                  <div className="pred-bar-wrap">
                    <div className="pred-bar" style={{ width: `${p.confidence}%` }} />
                  </div>
                </div>
                <span className="pred-conf">{p.confidence}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Treatment */}
        <div className="card treatment-card">
          <h3>Treatment Recommendation</h3>
          <div className="severity-tag" style={{ background: severityColor[treatment.severity] }}>
            Severity: {treatment.severity.toUpperCase()}
          </div>
          <p className="treatment-desc">{treatment.description}</p>
          <ul className="action-list">
            {treatment.actions.map((action, i) => (
              <li key={i}>
                <span className="action-bullet">→</span> {action}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

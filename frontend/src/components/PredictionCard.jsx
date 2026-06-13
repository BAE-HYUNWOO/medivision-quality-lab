export default function PredictionCard({ result }) {
  if (!result) return null;

  const probabilityRows = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);

  return (
    <section className="card prediction-card">
      <h2>Prediction</h2>
      <div className="prediction-main">
        <div>
          <span className="eyebrow">Predicted class</span>
          <strong>{result.predicted_label}</strong>
        </div>
        <div>
          <span className="eyebrow">Confidence</span>
          <strong>{Math.round(result.confidence * 1000) / 10}%</strong>
        </div>
      </div>

      <div className="probability-list">
        {probabilityRows.map(([label, value]) => (
          <div className="probability-row" key={label}>
            <span>{label}</span>
            <div className="probability-bar-wrap">
              <div className="probability-bar" style={{ width: `${Math.max(value * 100, 2)}%` }} />
            </div>
            <small>{Math.round(value * 1000) / 10}%</small>
          </div>
        ))}
      </div>
    </section>
  );
}

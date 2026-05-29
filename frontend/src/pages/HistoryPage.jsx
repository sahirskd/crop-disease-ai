import { useState, useEffect } from "react";
import { getHistory } from "../utils/api";

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory().then(setHistory).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page"><div className="spinner-large" /></div>;

  return (
    <div className="page history-page">
      <h2>Prediction History</h2>
      {history.length === 0 ? (
        <p className="empty-state">No predictions yet. Upload a leaf image to get started.</p>
      ) : (
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>Plant</th>
                <th>Disease</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.id}>
                  <td>{item.plant}</td>
                  <td>{item.disease}</td>
                  <td>{item.confidence}%</td>
                  <td>
                    <span className={`status-pill ${item.is_healthy ? "healthy" : "diseased"}`}>
                      {item.is_healthy ? "Healthy" : "Diseased"}
                    </span>
                  </td>
                  <td>{new Date(item.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

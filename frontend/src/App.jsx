import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import HistoryPage from "./pages/HistoryPage";
import Navbar from "./components/Navbar";
import ErrorBoundary from "./components/ErrorBoundary";

export default function App() {
  const [page, setPage] = useState("upload"); // "upload" | "result" | "history"
  const [result, setResult] = useState(null);

  const handleResult = (data) => {
    setResult(data);
    setPage("result");
  };

  return (
    <ErrorBoundary>
      <div className="app">
        <Navbar page={page} setPage={setPage} />
        <main>
          {page === "upload" && <UploadPage onResult={handleResult} />}
          {page === "result" && <ResultPage result={result} onBack={() => setPage("upload")} />}
          {page === "history" && <HistoryPage />}
        </main>
      </div>
    </ErrorBoundary>
  );
}

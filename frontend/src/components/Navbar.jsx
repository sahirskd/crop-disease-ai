export default function Navbar({ page, setPage }) {
  return (
    <nav className="navbar">
      <div className="nav-brand" onClick={() => setPage("upload")}>
        🌿 CropSense AI
      </div>
      <div className="nav-links">
        <button className={`nav-link ${page === "upload" ? "active" : ""}`} onClick={() => setPage("upload")}>
          Analyze
        </button>
        <button className={`nav-link ${page === "history" ? "active" : ""}`} onClick={() => setPage("history")}>
          History
        </button>
      </div>
    </nav>
  );
}

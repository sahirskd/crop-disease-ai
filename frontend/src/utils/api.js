const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

export async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    if (res.status === 429) {
      throw new Error("Too many requests. Please wait a moment and try again.");
    }
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function getDiseases() {
  const res = await fetch(`${BASE_URL}/diseases`);
  return res.json();
}

export async function getHistory(limit = 20) {
  const res = await fetch(`${BASE_URL}/history?limit=${limit}`);
  return res.json();
}

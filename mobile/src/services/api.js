// src/services/api.js
// All communication with the FastAPI backend lives here.

export const BASE_URL = "http://localhost:8000";

/** POST a camera frame (base64 URI) for full analysis */
export async function analyzeFrame(photoUri) {
  const formData = new FormData();

  // Web: photoUri is a blob URL, mobile: it's a file URI
  if (photoUri.startsWith("blob:") || photoUri.startsWith("data:")) {
    const blob = await fetch(photoUri).then(r => r.blob());
    formData.append("image", blob, "frame.jpg");
  } else {
    formData.append("image", {
      uri: photoUri,
      type: "image/jpeg",
      name: "frame.jpg",
    });
  }
  formData.append("mode", "describe");

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Analyze failed: ${res.status}`);
  return res.json();
  // → { description, objects, faces, alerts, timestamp }
}

/** POST a voice command string, get back a spoken response */
export async function sendCommand(command) {
  const res = await fetch(`${BASE_URL}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
  if (!res.ok) throw new Error(`Command failed: ${res.status}`);
  return res.json();
  // → { response: "..." }
}

/** POST a frame for OCR text extraction */
export async function extractText(photoUri) {
  const formData = new FormData();

  if (photoUri.startsWith("blob:") || photoUri.startsWith("data:")) {
    const blob = await fetch(photoUri).then(r => r.blob());
    formData.append("image", blob, "frame.jpg");
  } else {
    formData.append("image", {
      uri: photoUri,
      type: "image/jpeg",
      name: "frame.jpg",
    });
  }

  const res = await fetch(`${BASE_URL}/ocr`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`OCR failed: ${res.status}`);
  return res.json();
  // → { text: "..." | null }
}

/** GET scene memory summary */
export async function getMemory() {
  const res = await fetch(`${BASE_URL}/memory`);
  if (!res.ok) throw new Error(`Memory fetch failed: ${res.status}`);
  return res.json();
  // → { memory: { label: { count, last_seen, first_seen } } }
}

/** DELETE / clear memory */
export async function clearMemory() {
  const res = await fetch(`${BASE_URL}/memory`, { method: "DELETE" });
  return res.json();
}

/** Health check */
export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
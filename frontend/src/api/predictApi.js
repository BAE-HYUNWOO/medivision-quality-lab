const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function postImage(endpoint, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export function predictImage(file) {
  return postImage("/api/predict", file);
}

export function predictImageWithGradCam(file) {
  return postImage("/api/predict-with-gradcam", file);
}

export { API_BASE_URL };

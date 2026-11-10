const STORAGE_KEY = "kernel-path-progress-v1";

export function loadProgress(fallback = {}) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return { ...fallback, ...saved };
  } catch (error) {
    console.error("Failed to parse saved progress", error);
    return { ...fallback };
  }
}

export function saveProgress(progress) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
}


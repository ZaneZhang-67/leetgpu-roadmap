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

export function exportProgress(progress) {
  const blob = new Blob([JSON.stringify(progress, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `leetgpu-progress-${new Date().toISOString().slice(0, 10)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function importProgress(file) {
  const text = await file.text();
  const value = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("进度文件必须是 JSON 对象");
  }
  return value;
}

export function activityCounts(progress) {
  return Object.values(progress).reduce((counts, item) => {
    if (item?.status !== "done" || !item.completedAt) return counts;
    counts[item.completedAt] = (counts[item.completedAt] || 0) + 1;
    return counts;
  }, {});
}

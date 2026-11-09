const DAY_MS = 24 * 60 * 60 * 1000;
const COLORS = ["#17211f", "#12452f", "#157343", "#20a95a", "#42dc7c"];

function iso(date) {
  return date.toISOString().slice(0, 10);
}

function startOnSunday(date) {
  const value = new Date(date);
  value.setHours(12, 0, 0, 0);
  value.setDate(value.getDate() - value.getDay());
  return value;
}

export function renderHeatmap(container, counts) {
  const end = startOnSunday(new Date());
  end.setDate(end.getDate() + 6);
  const start = new Date(end.getTime() - 52 * 7 * DAY_MS + DAY_MS);
  const weeks = [];

  for (let week = 0; week < 52; week += 1) {
    const days = [];
    for (let day = 0; day < 7; day += 1) {
      const current = new Date(start.getTime() + (week * 7 + day) * DAY_MS);
      const key = iso(current);
      days.push({ date: key, count: counts[key] || 0 });
    }
    weeks.push(days);
  }

  const monthLabels = [];
  let lastMonth = -1;
  weeks.forEach((week, index) => {
    const month = new Date(`${week[0].date}T12:00:00`).getMonth();
    if (month !== lastMonth) {
      monthLabels.push({ index, label: new Date(`${week[0].date}T12:00:00`).toLocaleString("zh-CN", { month: "short" }) });
      lastMonth = month;
    }
  });

  container.innerHTML = `
    <div class="heatmap-months">
      ${monthLabels.map(({ index, label }) => `<span style="grid-column:${index + 1}">${label}</span>`).join("")}
    </div>
    <div class="heatmap-body">
      <div class="heatmap-days"><span>日</span><span>二</span><span>四</span><span>六</span></div>
      <div class="heatmap-grid" aria-label="LeetGPU 刷题热力图">
        ${weeks.flatMap((week) => week.map(({ date, count }) => {
          const level = Math.min(4, count);
          return `<button class="heat-cell" type="button" style="--cell-color:${COLORS[level]}" title="${date}：完成 ${count} 题" aria-label="${date} 完成 ${count} 题"></button>`;
        })).join("")}
      </div>
    </div>
    <div class="heatmap-legend"><span>少</span>${COLORS.map((color) => `<i style="--cell-color:${color}"></i>`).join("")}<span>多</span></div>
  `;
}

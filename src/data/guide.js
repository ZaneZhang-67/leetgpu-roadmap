export const principles = [
  {
    title: "先写对",
    text: "先覆盖任意尺寸、边界和数据类型；通过测试只是第一步，但它必须是第一步。",
  },
  {
    title: "再解释",
    text: "每题都要能说清线程负责什么、地址怎样算、同步为什么出现在这里。",
  },
  {
    title: "最后优化",
    text: "一次只改变一个因素，用相同 GPU 和 shape 比较结果，保留基线与实验记录。",
  },
];

export const weeklyPlan = [
  { day: "Day 1", task: "读题并手画数据布局", output: "写出 CPU 公式和索引公式" },
  { day: "Day 2", task: "独立写基础 kernel", output: "通过全部正确性测试" },
  { day: "Day 3", task: "解释线程与访存", output: "完成题解笔记" },
  { day: "Day 4", task: "实现一个变式", output: "支持不同布局或数据类型" },
  { day: "Day 5", task: "做一次性能实验", output: "记录条件、结果与结论" },
  { day: "Weekend", task: "关掉旧代码重写", output: "确认可以从空白复现" },
];

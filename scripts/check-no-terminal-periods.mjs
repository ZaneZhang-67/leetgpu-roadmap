import { readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";

const roots = ["src", "README.md", "index.html"];
const extensions = new Set([".js", ".md", ".html"]);
const violations = [];

function inspect(path) {
  // Generated files preserve CMS article text; check only hand-written source here.
  if (path.split(/[\\/]/).includes("generated")) return;
  if (statSync(path).isDirectory()) {
    for (const name of readdirSync(path)) inspect(join(path, name));
    return;
  }
  if (!extensions.has(extname(path))) return;

  const lines = readFileSync(path, "utf8").split("\n");
  lines.forEach((line, index) => {
    if (line.includes("。")) {
      violations.push(`${relative(process.cwd(), path)}:${index + 1}: ${line.trim()}`);
    }
  });
}

for (const root of roots) inspect(root);

if (violations.length) {
  console.error("发现句末或段末中文句号，请移除：\n" + violations.join("\n"));
  process.exit(1);
}

console.log("No terminal Chinese periods found");

const { spawn } = require("child_process");
const path = require("path");

const mode = process.argv[2] || "run";
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const cypressBin = path.join(__dirname, "..", "node_modules", "cypress", "bin", "cypress");
const child = spawn(process.execPath, [cypressBin, mode, ...process.argv.slice(3)], {
  stdio: "inherit",
  env,
});

child.on("exit", (code) => process.exit(code || 0));


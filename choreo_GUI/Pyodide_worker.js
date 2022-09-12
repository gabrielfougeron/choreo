// load pyodide.js
importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.2/full/pyodide.js");


async function loadPyodideAndPackages() {
  self.pyodide = await loadPyodide();
  await pyodide.loadPackage([
    "matplotlib",
    "sparseqr",
    "networkx",
    "./python_dist/choreo-0.1.0-cp310-cp310-emscripten_3_1_14_wasm32.whl"
  ]);
}
let pyodideReadyPromise = loadPyodideAndPackages();

self.onmessage = function(message) {
    console.log(message.data);
  }
// load pyodide.js
importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.2/full/pyodide.js");
importScripts("https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.0/FileSaver.min.js")


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
  
    if ((typeof message.data.funname != "undefined") && (typeof message.data.args != "undefined")) {

        console.log("Attempting to execute function",message.data.funname,"with arguments",message.data.args);

        self[message.data.funname](message.data.args);

    } else {

        console.log('WebWorker could not resolve message :',message);

    }
  }

self.LoadDataInWorker = function(datadict) {
    
    for (const [key, value] of Object.entries(datadict)) {
        self[key] = value;
    }

}
  
// python_cache_behavior = {}
python_cache_behavior = {cache: "no-cache"}
  
self.ExecutePythonFile = function(filename) {
    let load_txt = fetch(filename,python_cache_behavior) ; 
    load_txt.then(function(response) {
        return response.text();
    }).then(async function(text) {  
        await pyodideReadyPromise; 
        txt = pyodide.runPython(text);
    });
}

function RedirectPythonPrint(txt) {

    // console.log(txt);

    self.postMessage({
        funname : "PythonPrint",
        args    : {
                "txt":txt,
            }
        }
    )

}

var AskForNext;

// load pyodide.js
importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");
// importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js");
// importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.0/full/pyodide.js");
// importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.2/full/pyodide.js");
// importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.3/full/pyodide.js");
// importScripts("https://cdn.jsdelivr.net/pyodide/dev/full/pyodide.js");

async function loadPyodideAndPackages() {

    RedirectPythonPrint("Starting python initialization ...\n")								

    self.pyodide = await loadPyodide({
        stdout: RedirectPythonPrint,
        stderr: RedirectPythonPrint,
    })

    RedirectPythonPrint("Python initialized\n")		

    RedirectPythonPrint("Importing packages ...")
    
    await pyodide.loadPackage([
        // "micropip",
        "matplotlib",
        "sparseqr",
        "networkx",
        "mpmath",
        "threadpoolctl",
        // "./python_dist/choreo-0.2.0-cp311-cp311-emscripten_3_1_45_wasm32.whl",
        "./python_dist/choreo-1.0.0-cp311-cp311-emscripten_3_1_45_wasm32.whl",
        "./python_dist/pyquickbench-0.2.1-py3-none-any.whl",
        "tqdm",
    ])

}

let pyodideReadyPromise = loadPyodideAndPackages()

self.onmessage = async function(message) {
  
    if ((typeof message.data.funname != "undefined") && (typeof message.data.args != "undefined")) {

        // console.log("Attempting to execute function",message.data.funname,"with arguments",message.data.args)

        const the_fun = self[message.data.funname]

        const isAsync = the_fun.constructor.name === "AsyncFunction"

        if (isAsync) {
            await the_fun(message.data.args)
        } else {
            the_fun(message.data.args)
        }


    } else {

        console.log('WebWorker could not resolve message :',message)

    }
  }

self.LoadDataInWorker = function(datadict) {
    
    for (const [key, value] of Object.entries(datadict)) {
        self[key] = value;
        // console.log(key,value)

    }

}
  
var python_cache_behavior = {}
// var python_cache_behavior = {cache: "no-cache"}
  
self.ExecutePythonFile = function(filename) {
    let load_txt = fetch(filename,python_cache_behavior) ; 
    load_txt.then(function(response) {
        return response.text();
    }).then(async function(text) {  
        await pyodideReadyPromise; 
        txt = pyodide.runPython(text);
    });
}

self.setAskForNextBuffer = function(Buffer) {

    AskForNext = new Uint8Array(Buffer);

}

async function syncFromDisk() {
    return await new Promise((resolve, _) => pyodide.FS.syncfs(true, resolve));
 }

var NativeFS
var NativeFSIsSetUp = false

self.SetupWorkspaceInWorker = async function(dirHandle) {

    if (NativeFSIsSetUp) {

        syncFromDisk()

    } else {

        await pyodideReadyPromise
        NativeFS = await pyodide.mountNativeFS("/Workspace", dirHandle)

        NativeFSIsSetUp = true

    }

}
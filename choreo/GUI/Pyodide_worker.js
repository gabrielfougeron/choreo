
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
importScripts("assets/pyodide/pyodide.js");

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
        "networkx",
        "mpmath",
        "scipy",
        "threadpoolctl",
        "./python_dist/choreo-1.0.0-cp312-cp312-pyodide_2024_0_wasm32.whl",
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
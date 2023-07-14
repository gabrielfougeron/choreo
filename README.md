# choreo
Finds periodic solutions to the gravitational N-body problem. 

## Try out this project, no installation required!

Check out the online in-browser GUI: https://gabrielfougeron.github.io/choreo/

## Install the package with pip

The package is not available on PyPA yet, but will be in the future.
Till then, the installation process is the following:

 - Download this project. For instance using git: `git clone git@github.com:gabrielfougeron/choreo.git`
 - Open the directory: `cd choreo`
 - Build and install using pip: `pip install .`

## Build wheel for pyodide

After sourcing emsdk environment, run the following:

```
pyodide build && cp ./dist/choreo-0.2.0-cp311-cp311-emscripten_3_1_32_wasm32.whl ./choreo_GUI/python_dist/choreo-0.2.0-cp311-cp311-emscripten_3_1_32_wasm32.whl
```

## Power up the GUI solver with the CLI backend
Using clang or gcc as a C compiler, the single-threaded CLI solver is about 3 times faster that the wasm in-browser GUI solver. In addition, several independent single-threaded solvers can be launched simultaneously using a single command.

To use the CLI backend, follow these steps:

- Install the package
- In the GUI, define a workspace folder under `Play => Workspace => Setup Workspace`
- Every time the workspace is reloaded under `Play => Workspace => Reload Workspace` **or** every time a new initial state is generated in the GUI, a new configuration file `choreo_config.json` is written to disk.
- In the command line, run the installed script as `choreo_GUI_in_CLI -f /path/to/workspace/folder/` 

## Online documentation

Available at: https://gabrielfougeron.github.io/choreo-docs/

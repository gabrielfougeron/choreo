# Choreographies2
Finds periodic solutions to the N-body problem

## List of dependencies :
  
  - The required conda environment is described in the file choreo_env.yml
  - To set up, run the following :
    
```
conda env create -f choreo_env.yml

conda activate choreo
```

## Inplace compilation of Cython code

Run the following :

```
python setup.py build_ext --inplace
```

## Build wheel package

Run the following :

```
python setup.py bdist_wheel
```

## Build wheel for pyodide

After sourcing emsdk environment, run the following:

```
pyodide build && cp ./dist/choreo-0.1.0-cp310-cp310-emscripten_3_1_14_wasm32.whl ./choreo_GUI/python_dist/choreo-0.1.0-cp310-cp310-emscripten_3_1_14_wasm32.whl 
```

## Want to start finding new solutions, no installation needed ?

Check out the online in-browser GUI: https://gabrielfougeron.github.io/choreo/

## Power up the GUI solver with the CLI backend
Using clang or gcc as a C compiler, the single-threaded CLI solver is about 3 times faster that the wasm in-browser GUI solver. Several independant single-threaded solvers can be launched simultaneously using a single command.

- Clone this repository.
- Create and activate the correct conda environment.
- Compile the Cython part of the library.
- In the GUI, setup a Workspace on your computer.
- In `./scripts/Load_GUI_params_and_run.py`, set the variable `Workspace_folder` to the path of your Workspace.
- Execute the script : `python ./scripts/Load_GUI_params_and_run.py`

## Online documentation

Available at: https://gabrielfougeron.github.io/choreo-docs/

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

Run the following:

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
pyodide build && cp ./dist/choreo-0.2.0-cp310-cp310-emscripten_3_1_27_wasm32.whl ./choreo_GUI/python_dist/choreo-0.2.0-cp310-cp310-emscripten_3_1_27_wasm32.whl 
```

## Want to start finding new solutions ?

Simply run the following script.

```
python Choreo_sniffall.py
```

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

## Want to start finding new solutions ?

Simply run the following script :

```
python Choreo_sniffall.py
```

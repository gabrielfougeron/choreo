# Choreographies2
Finds periodic solutions to the N-body problem

## List of dependencies :
  
  - The required conda environment is described in the file choreo_env.yml
  - To set up, run the following :
    
```
conda env create -f choreo_env.yml

conda activate choreo
```

## Compilation of Cython code

Run the following :

```
python setup.py build_ext --inplace
```

## Want to start finding new solutions ?

Simply run the follwoing scipt :

```
python Choreo_sniffall.py
```

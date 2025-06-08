# Particle_Filter
MATLAB implementation of standard particle filter, auxiliary particle filter, mixture particle filter, and out-of-sequence particle filter for terrain-referenced navigation.

This repository now also includes an experimental **Python** version of the basic particle filter located in `python/`.  The Python code mirrors the MATLAB example using NumPy and SciPy for interpolation and filtering utilities.

How to run:

run main_OOSM.m

For the Python prototype:

```bash
python -m python.pf_main
```


Please cite the following paper if you find this code helpful:

Youngjoo Kim et al., "Utilizing Out-of-Sequnece Measurement for Ambiguous Update in Particle Filtering", IEEE Transactions on Aerospace and Electronic Systems, 54(1), 2018. Available: https://www.researchgate.net/publication/319193705_Utilizing_Out-of-Sequence_Measurement_for_Ambiguous_Update_in_Particle_Filtering

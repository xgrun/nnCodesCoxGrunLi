This is the code used in the paper by Cox, Grundler, and Li.

SL stands for Scikit-Learn, and TF stands for TensorFlow. Emulation is for the models which take the input parameters X and K to predict the observables v1 and v2. Inversion is for models which infer X and K from v1 and v2.

xytrain90.dat holds 89 data points from an IBUU simulation of Au+Au collisions at beam energies of 1.23A GeV. This data is used to train and test all models.

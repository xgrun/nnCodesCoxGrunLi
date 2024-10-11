Users of this code should cite the following publication: N. Cox, X. Grundler, B.A. Li, Neural network emulation of flow in heavy-ion collisions at intermediate energies. Phys. Rev. C **110**, 044604 (2024). [doi:10.1103/PhysRevC.110.044604](http://dx.doi.org/10.1103/PhysRevC.110.044604)

SL stands for Scikit-Learn, and TF stands for TensorFlow. Emulation is for the models which take the input parameters X and K to predict the observables F1 and v2. Inversion is for models which infer X and K from F1 and v2.

xytrain90.dat holds 89 data points from an IBUU simulation of Au+Au collisions at beam energies of 1.23A GeV. This data is used to train and test all models.

# ACS-SD

This repository contains the implementation of ACS-SD algorithm proposed in the article ACS-SD: Ant Colony System for Subgroup Discovery.

# 1. Project Scripts
The ./ACS-SD.py file contains the implementation of the ACS-SD algorithm.
The ./constants.py file contains variable definitions.
The ./fase1.py file contains the implementation of the Initial seed generation phase, including Pareto filter, clustering, and dynamic ant allocation.
The ./functions.py file contains auxiliary method definitions.

# 2. Datasets
The datasets directory contains the datasets used to validate and analyse the algorithm.

# Execution
When running the ACS-SD algorithm, the following considerations should be taken into account:

* The variable to be used as the target must have the value "Class", unless this is modified in the source code.
* The algorithm can be executed either for a specific value of the target variable or in dynamic mode, where the algorithm automatically determines which target value to consider. This behavior is controlled by the value of the `SINGLE_TARGET` variable in the `constants.py` file (`False` enables dynamic mode).


# Corresponding Authors
For additional information, please contact the author via email: aaron.garcia.muniz@upm.es

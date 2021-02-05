# Muscle Synergies

README.md em português: [README.pb.md](README.pb.md)

Extract muscle synergies from experimental data obtained using Vicon Nexus.

## Contents

- [Project status](#project-status)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Project status

Currently, loading the data from the CSV file is being worked on.  There are
some tests implemented using `pytest` but the internal API was changed a bit and
the tests will need to be refactored.  Also because I'm still finishing the
implementation of this functionality, the docstrings are incomplete.

After that first part of the project is finished, the EMG analysis based on
non-negative matrix factorisation, a method recently shown to be superior to
alternatives [[1]](#1), will be implemented.

## Acknowledgements

This analysis is being developed for my internship under the supervision of
professor Heiliane de Brito at the [Department of Morphological
Sciences](https://mor.ccb.ufsc.br/) in the Federal University of Santa Catarina
(UFSC).

## References
<a id="1">[1]</a>
Rabbi, M.F., Pizzolato, C., Lloyd, D.G. et al. Non-negative matrix factorisation is the most appropriate method for extraction of muscle synergies in walking and running.  
Sci Rep 10, 8266 (2020).  
https://doi.org/10.1038/s41598-020-65257-w

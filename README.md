# Muscle Synergies

Versão desse documento em português: [portuguese](README.pb.md)

Extract muscle synergies from experimental data obtained using Vicon Nexus.

## Contents

- [Project status](#project-status)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Project status

The project is being actively developed. Currently, loading the data from the
CSV file is possible as demonstrated in an
[example notebook](examples/1.%20Loading%20and%20plotting%20data.ipynb).
The goal now is to analyse the EMG data to
determine muscle synergies using the method in [[1]](#1). I'll also simultaneously
fill the docstrings (most are missing) and work on general code and repo
quality.

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

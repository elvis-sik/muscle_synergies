# Muscle Synergies

English version of this document: [inglês](README.md)

Extração de sinergias musculares a partir de dados experimentais obtidos usando
Vicon Nexus.

## Conteúdo

- [Status do projeto](#status-do-projeto)
- [Agradecimentos](#agradecimentos)
- [Referências](#referências)

## Status do projeto

No momento, estou trabalhando no código que carrega os dados do arquivo CSV.
Tem testes implementados usando `pytest` mas a API interna mudou um pouco e os
testes precisam ser refatorados.  Também pelo fato de ainda estar terminando de
implementar essa funcionalidade, as docstrings estão incompletas.

Depois que essa primeira parte do projeto estiver terminada, a análise de EMG
baseada em fatoração de matrizes não-negatives, método que foi recentemente
demonstrado superior a outros [[1]](#1), será implementada.

## Agradecimentos

Essa análise está sendo desenvolvida para meu estágio sob a supervisão da
professora Heiliane de Brito no [Departamento de Ciências
Morfológicas](https://mor.ccb.ufsc.br/) na Universidade Federal de Santa
Catarina (UFSC).

## Referências
<a id="1">[1]</a> Rabbi, M.F., Pizzolato, C., Lloyd, D.G. et al. Non-negative matrix factorisation is the most appropriate method for extraction of muscle synergies in walking and running.  
Sci Rep 10, 8266 (2020).  
https://doi.org/10.1038/s41598-020-65257-w

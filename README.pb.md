# Muscle Synergies

English version of this document: [inglês](README.md)

Extração de sinergias musculares a partir de dados experimentais obtidos usando
Vicon Nexus.

## Conteúdo

- [Status do projeto](#status-do-projeto)
- [Agradecimentos](#agradecimentos)
- [Referências](#referências)

## Status do projeto

O projeto está sendo ativamente desenvolvido. No momento, carregar os dados do
arquivo CSV é possível conforme demonstrado em um  [notebook de
exemplo](examples/1. Loading and plotting data.ipynb).  O objetivo agora é
analisar os dados EMG para determinar as sinergias musculares usando o método em
[[1]](#1). Vou também simultaneamente preencher as docstrings (a maioria está
faltando) e trabalhar de forma geral na qualidade do código e do repo.

## Agradecimentos

Essa análise está sendo desenvolvida para meu estágio sob a supervisão da
professora Heiliane de Brito no [Departamento de Ciências
Morfológicas](https://mor.ccb.ufsc.br/) na Universidade Federal de Santa
Catarina (UFSC).

## Referências
<a id="1">[1]</a> Rabbi, M.F., Pizzolato, C., Lloyd, D.G. et al. Non-negative matrix factorisation is the most appropriate method for extraction of muscle synergies in walking and running.  
Sci Rep 10, 8266 (2020).  
https://doi.org/10.1038/s41598-020-65257-w

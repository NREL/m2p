[![Build Status](https://travis-ci.com/NREL/m2p.svg?branch=master)](https://travis-ci.com/NREL/m2p)
[![GitHub version](https://badge.fury.io/gh/NREL%2Fm2p.svg)](https://badge.fury.io/gh/NREL%2Fm2p)

# Monomers to Polymers (m2p)

A simple interface for converting monomers to polymers using SMILES representation.

## Related Work

1. [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
2. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
3. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
4. [Neural Message Passing with Edge Updates for Predicting Properties of Molecules and Materials](https://arxiv.org/abs/1806.03146)

## (Main) Requirements

Rdkit install can be performed per the following information.
- [rdkit](http://www.rdkit.org/docs/Install.html)

To install rdkit via conda, use:
```conda install -c rdkit rdkit```

## Getting started

The library uses known reaction chemistries to build polymer chains from monomers. The polymer chemistries available include vinyls, acrylates, esters, amides, imides, and carbonates.

The library can generate multiple replicate structures to create polymer chains represented at the atom and bond level. The chains can be any degree of polymerization (DP). RDKit reaction smarts are used to manipulate the molecular structures and perform *in silico* reactions.

Muscle Synergies
################

Determine muscle synergies on the data outputted by the Vicon Nexus machine.

.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:

   install/index
   tutorials/index
   api/index
   dev/index
   references/index

Spatial muscle synergies are a way to model muscle activation with fewer dimensions :cite:p:`dAvella2009`. If :math:`m(t) \in \mathbb R^L` is the vector with the activation of each one of :math:`L` muscles at time :math:`t`, we would look for :math:`K` synergy components :math:`w_i` such that

.. math:: m(t) \approx \sum_{i = 1}^K c_i(t) w_i

Where :math:`c_i(t) \in \mathbb R` is the coefficient multiplying synergy component :math:`w_i` at time :math:`t`.

A common way to find the synergy components :math:`w_i` and the coefficients :math:`c_i(t)` is to start with these preprocessing steps:

  1. Zero-center the signal.
  2. Starting from the raw EMG signal, compute either its linear envelope or its RMS to get a measure of its instantaneous amplitude.
  3. Time normalize the signal down to a :math:`N` (with :math:`N = 200` being a common value) time points.
  4. Amplitude normalize the signal.

The signal is now in a matrix :math:`M \in \mathbb R^{N \times L}`. Standard dimensionality reduction techniques can be used to find the synergy components. For example, in non-negative matrix factorization (found to be superior to other methods :cite:p:`Rabbi2020`) we would look for matrices :math:`H \in \mathbb R^{N \times K}` and :math:`W \in \mathbb R^{K \times L}` such that:

.. math:: M \approx H W

with :math:`M`, :math:`H` and :math:`W` having only nonnegative entries. In this formulation, the rows of :math:`W` would be the synergy components. Note that many papers adopt a different notation, including :cite:t:`Rabbi2020`.

This package offers the following:

  + a function that loads and parses the CSV data file outputted by Vicon Nexus.
  + wrappers around standard functions in the Python ecosystem to make them friendlier for users who want to use them to do synergy analysis on EMG datasets.

.. _api-reference-label:

API Reference
=============

The `API Reference <http://muscle_synergies.readthedocs.io>`_ provides API-level documentation.

.. _report-bugs-label:

Report Bugs
===========

Report bugs at the `issue tracker <https://github.com/elvis-sik/muscle_synergies/issues>`_.

Please include:

  - Operating system name and version.
  - Any details about your local setup that might be helpful in troubleshooting.
  - Detailed steps to reproduce the bug.


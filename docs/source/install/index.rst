Getting Started
###############

.. _install-guide-label:

Quick Start
===========

Install the package:

.. code-block:: console

    $ pip install muscle_synergies

Load a dataset from a Vicon Nexus file:

.. code-block:: python

    import muscle_synergies as ms
    vicon_data = ms.load_vicon_data('vicon-nexus-output.csv')

Then get a :py:class:`pandas.DataFrame` containing the EMG data:

.. code-block:: python

    emg_df = vicon_data.emg.df

Plot it:

.. code-block:: python

    ms.plot_signal(emg_df)

See the tutorials for more.

For users new to Python
=======================

Using Google Colab is recommended because it requires no installation.

Using Google Colab (recommended)
--------------------------------

An easy way to start is to use `Google Colaboratory <https://colab.research.google.com/>`_.
In the first cell run the following command to install the `muscle_synergies` package:

.. code-block:: console

    !pip install muscle_synergies

Then download the provided example dataset running the following command:

.. code-block:: console

    !wget https://github.com/elvis-sik/muscle_synergies/raw/master/sample_data/dynamic_trial.csv

The user then may load the dataset with the following command:

.. code-block:: python

   from muscle_synergies import load_vicon_file
   load_vicon_file('dynamic_trial.csv')

And continue following along the :ref:`tutorials-label`.

Using Anaconda
--------------

For local development (i.e., in the user's machine instead of the cloud service provided by Google), installing the `Anaconda distribution <https://docs.anaconda.com/anaconda/install/>`_ and then using `Jupyter Notebook <https://jupyter.org/index.html>`_ (which already comes installed with Anaconda) is recommended.

Learning Python
---------------

Doing anything non-trivial with this package will benefit from some Python knowledge. Even learning just a little bit might go a long way towards making everything easier.
For users who are interested in following this route, introductory materials focusing on scientific computing or data analysis would be recommended.
Specifically, getting some exposure to `NumPy`, `Pandas` and `Matplotlib` would be useful.

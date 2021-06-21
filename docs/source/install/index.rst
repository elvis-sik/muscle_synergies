Getting Started
###############

.. _install-guide-label:

Quick Start
===========

Install the package with:

.. code-block:: console

    $ pip install muscle_synergies

Load a dataset from a Vicon Nexus file with:

.. code-block:: python

    import muscle_synergies as ms
    vicon_data = ms.load_vicon_data('vicon-nexus-output.csv')

Then get a :py:class:`pandas.DataFrame` containing the EMG data with:

.. code-block:: python

    emg_df = vicon_data.emg.df

Plot it with:

.. code-block:: python

    ms.plot_signal(emg_df)

See the tutorials for more detailed presentation and for how to find synergies.

For users new to Python
=======================

Using Google Colab is recommended because it is easy to start since it requires no installation. The user is encouraged to follow along the tutorials. Also, going through some course or tutorial introducing Python will likely make using this package easier.

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

And continue following along the tutorials.

Using Anaconda
--------------

For local development (i.e., in the user's machine instead of the cloud service provided by Google), installing the `Anaconda distribution <https://docs.anaconda.com/anaconda/install/>`_ and then using `Jupyter Notebook <https://jupyter.org/index.html>`_ (which already comes installed with Anaconda) is recommended.

Learning Python
---------------

If the user wants to learn a bit of Python (likely needed to do anything non-trivial with this package), I recommend using a tutorial focused on scientific computing or data analysis.
Ideally, they would get some exposure to `NumPy`, `Pandas` and `Matplotlib`.
Googling around should easily bring a plethora of good materials.
Just for the sake of completeness, the following course seems to include all of the basics needed: https://www.edx.org/course/python-basics-for-data-science. I have not checked it with any rigor though.

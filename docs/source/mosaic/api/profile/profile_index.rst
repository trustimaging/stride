==================
Profiling
==================

PLEASE NOTE: This is a beta feature that is still under development.

Profiling of Mosaic applications can be activated by passing ``profile=True`` as an argument in interactive mode:

.. code-block:: python

    mosaic.interactive('on', profile=True)

or by using the flag ``--profile`` when starting the runtime from the command line:

.. code-block:: shell

    mrun --profile python script.py

After running the Mosaic application, a file will have been generated with the form ``<date>-<time>.profile.h5`` in
the working directory. To visualise the resulting profile, you can run:

.. code-block:: shell

    mprof <date>-<time>.profile.h5

and access ``http://127.0.0.1:8050/`` in your browser. This will show you a timeline of the Stride execution.

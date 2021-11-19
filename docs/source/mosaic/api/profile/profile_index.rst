==================
Profiling
==================

Profiling of Mosaic applications can be activated by passing ``profile=True`` as an argument in interactive mode:

.. code-block:: python

    mosaic.interactive('on', profile=True)

or by using the flag ``--profile`` when starting the runtime from the command line:

.. code-block:: shell

    mrun --profile python script.py

Then, when ready to start the profiling, start a profiling trace in your application by running:

.. code-block:: python

    from mosaic.profile import profiler

    profiler.start_trace()  # profiling starts from this point onwards

After running the Mosaic application, a series of files will have been generated within a folder of the form ``<date>-<time>.profile.parts`` in
working directory. To visualise the resulting profile, you can run:

.. code-block:: shell

    mprof <date>-<time>.profile.parts

and access ``http://127.0.0.1:8050/`` in your browser.

To prevent a block of code from being profiled, you can use the context manager ``no_profiler``:

.. code-block:: python

    from mosaic.profile import no_profiler

    ...code being profiled

    with no_profiler():
        ...code not profiled

    ...code being profiled again

You can also prevent a function from being profiled with the decorator ``@skip_profile``:

.. code-block:: python

    from mosaic.profile import skip_profile

    def function_being_profiled():
        pass

    @skip_profile
    def function_not_profiled():
        pass


.. toctree::

    profile

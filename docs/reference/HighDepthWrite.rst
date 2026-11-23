.. py:module:: PIL.HighDepthWrite
.. py:currentmodule:: PIL.HighDepthWrite

:py:mod:`~PIL.HighDepthWrite` module
====================================

The :py:mod:`~PIL.HighDepthWrite` module saves high depth PNG, TIFF and BMP
files.  The decoded size is checked against an optional memory budget before
pixels are read, the encoder can be cancelled between blocks, and output
written to a path only replaces the destination once the whole image is
encoded.

Multi-page TIFF output is committed in the order the pages were supplied.

Functions
---------

.. autofunction:: save

Classes
-------

.. autoclass:: Cancel
    :members:

.. autoexception:: SaveError
.. autoexception:: SaveCancelled

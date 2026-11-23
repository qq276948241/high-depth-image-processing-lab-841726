.. py:module:: PIL.HighDepth
.. py:currentmodule:: PIL.HighDepth

:py:mod:`~PIL.HighDepth` module
===============================

The :py:mod:`~PIL.HighDepth` module decodes PNG, TIFF and BMP data into full
precision sample planes.  The same bytes may be supplied as a path, a binary
file object, a bytes object or an iterable of chunks; all return identical
pixels and metadata.  Structural errors are raised before a pixel buffer is
allocated.

Example: read 16-bit samples
----------------------------

::

    from PIL import HighDepth

    image = HighDepth.decode("16bit.tif")
    print(image.info.bit_depth)
    print([image.sample(x, 0) for x in range(image.info.width)])

Functions
---------

.. autofunction:: decode
.. autofunction:: decode_planes
.. autofunction:: probe
.. autofunction:: read
.. autofunction:: rescale_plane
.. autofunction:: sample_to_unit
.. autofunction:: unit_to_sample

Classes
-------

.. autoclass:: DecodedImage
    :members:

.. autoclass:: ImageInfo
    :members:

.. autoclass:: MemoryBudget
    :members:

.. autoclass:: StreamReader
    :members:

.. autoexception:: DecodeError
.. autoexception:: MemoryBudgetError

Constants
---------

.. autodata:: MAX_DECODED_BYTES
.. autodata:: PNG_PASSES

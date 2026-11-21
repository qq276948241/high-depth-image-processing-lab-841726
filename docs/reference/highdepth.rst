.. py:module:: PIL.highdepth
.. py:currentmodule:: PIL.highdepth

:py:mod:`~PIL.highdepth` module
===============================

The :py:mod:`~PIL.highdepth` package adds a strict, lossless path for images
whose precision or palette data the classic :py:class:`~PIL.Image.Image` API
cannot fully carry.  It is additive: :mod:`PIL.Image`, :mod:`PIL.ImageFile`,
:mod:`PIL.ImageCms`, :class:`PIL.PngImagePlugin.PngImageFile`,
:class:`PIL.TiffImagePlugin.TiffImageFile` and
:class:`PIL.BmpImagePlugin.BmpImageFile` behave exactly as before.

Reading
-------

.. autofunction:: read
.. autofunction:: read_frames
.. autofunction:: iter_frames
.. autofunction:: read_header
.. autofunction:: stream_decode
.. autofunction:: iter_image_tiles

The normalized representation
-----------------------------

.. autoclass:: ImageData
    :members: describe, sha256, tobytes, row_samples, sample_at, iter_rows, to_pil, from_pil, to_rgba, flatten, validate
    :special-members: __init__

.. autoclass:: Palette
    :members: rgb_bytes, rgba_bytes, alpha_bytes, has_alpha, transparent_indices, with_entry, from_rgb, from_rgba
    :special-members: __init__

Quantization
------------

.. autofunction:: quantize
.. autofunction:: build_palette
.. autofunction:: map_pixels
.. autoclass:: QuantizeOptions
    :members:

Colour management
-----------------

.. autoclass:: ColorPipeline
    :members: apply, describe, digest
    :special-members: __init__
.. autofunction:: srgb_to_linear
.. autofunction:: linear_to_srgb
.. autofunction:: apply_gamma
.. autofunction:: load_profile
.. autofunction:: unpremultiply

Budgets, cancellation and streaming
-----------------------------------

.. autoclass:: MemoryBudget
    :members: reserve, release, check, used, peak, remaining
    :special-members: __init__
.. autoclass:: CancellationToken
    :members: cancel, cancelled, reset, raise_if_cancelled
.. autoclass:: Tile
    :members: width, height
.. autofunction:: iter_tiles
.. autofunction:: iter_rows

Batch processing
----------------

.. autofunction:: PIL.highdepth.pipeline.run_batch
.. autoclass:: PIL.highdepth.pipeline.Job
    :members:
.. autoclass:: PIL.highdepth.pipeline.JobResult
    :members:

Writing
-------

.. autofunction:: PIL.highdepth.encode.write_frames
.. autofunction:: PIL.highdepth.encode.verify_roundtrip

Serialization
-------------

.. autofunction:: dump
.. autofunction:: load

Errors
------

Every error below carries a stable ``code`` attribute and a documented base
class.  Format errors derive from :class:`ValueError`; resource errors from
:class:`MemoryError`.

.. autoexception:: HighDepthError
.. autoexception:: SourceError
.. autoexception:: FormatError
.. autoexception:: UnsupportedFeatureError
.. autoexception:: TruncatedDataError
.. autoexception:: ChunkLengthError
.. autoexception:: ChecksumError
.. autoexception:: TrailingDataError
.. autoexception:: IllegalPaletteError
.. autoexception:: DimensionOverflowError
.. autoexception:: BudgetExceededError
.. autoexception:: BitDepthError
.. autoexception:: EmptyImageError
.. autoexception:: InvalidParameterError
.. autoexception:: ProfileError
.. autoexception:: UnknownColorSpaceError
.. autoexception:: NonFiniteSampleError
.. autoexception:: CancelledError
.. autoexception:: WriteVerificationError

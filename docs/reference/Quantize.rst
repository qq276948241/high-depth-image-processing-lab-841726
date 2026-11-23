.. py:module:: PIL.Quantize
.. py:currentmodule:: PIL.Quantize

:py:mod:`~PIL.Quantize` module
==============================

The :py:mod:`~PIL.Quantize` module provides a deterministic quantizer for the
modes the C quantizer cannot handle: "I;16", "I;16L", "I;16B", "I;16N", "I",
"LA" and "PA".  It backs :py:meth:`~PIL.Image.Image.quantize` and the
``Palette.ADAPTIVE`` path of :py:meth:`~PIL.Image.Image.convert` for those
modes.

Palette ordering, colour distance, tie-breaking and Floyd-Steinberg rounding
are fixed, so repeated calls with the same input, palette size and arguments
produce identical bytes.

Functions
---------

.. autofunction:: quantize
.. autofunction:: quantize_to_palette
.. autofunction:: quantize_to_web_palette

Constants
---------

.. autodata:: HIGH_DEPTH_MODES
.. autodata:: MAX_QUANTIZE_COLORS

.. autoexception:: QuantizeError

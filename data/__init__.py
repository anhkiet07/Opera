"""Dataset package.

``DATA_ROOT`` is where raw / processed datasets live. It defaults to this
``data/`` directory inside the repository (so ``data/physionet``, ``data/P12``,
...) and can be overridden with the ``QUITE_DATA_ROOT`` environment variable.
"""
import os

DATA_ROOT = os.environ.get(
    "QUITE_DATA_ROOT", os.path.dirname(os.path.abspath(__file__))
)

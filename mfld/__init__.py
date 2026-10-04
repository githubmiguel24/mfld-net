"""MFLD-net: Mobile Fish Landmark Detection network.

Re-implementation of the architecture and training recipe described in:

    Saleh A., Jones D., Jerry D., Rahimi Azghadi M. (2023)
    "MFLD-net: a lightweight deep learning network for fish morphometry
    using landmark detection". Aquatic Ecology 57:913-931.
    https://doi.org/10.1007/s10452-023-10044-8

Sub-modules are imported explicitly (``from mfld.model import MFLDNet``) so
that the numpy-only parts (metrics, morphometry, heatmaps, synthetic data)
work without PyTorch installed.
"""

__version__ = "1.0.0"

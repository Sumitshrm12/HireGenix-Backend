# sitecustomize.py - Loaded automatically by Python at startup
# This file suppresses known warnings that don't affect functionality

import warnings

# Fix missing importlib.metadata.packages_distributions on Python 3.9 by
# borrowing the backport implementation if available.
try:
    import importlib.metadata as _stdlib_metadata
    try:
        from importlib_metadata import packages_distributions as _pkgs
    except Exception:
        _pkgs = None
    if _pkgs:
        _stdlib_metadata.packages_distributions = _pkgs
except Exception:
    # If anything goes wrong, leave the default behavior.
    pass

# Suppress urllib3 LibreSSL compatibility warning
# This warning occurs on macOS with older Python versions compiled against LibreSSL
# It doesn't affect functionality - HTTPS still works correctly
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Suppress NotOpenSSLWarning specifically
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

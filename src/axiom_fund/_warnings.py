"""Warning filter configuration for the axiom_fund package.

The wrds library (3.1.6) contains SyntaxWarnings in its own source code
(invalid escape sequences in string literals). These are harmless and the
WRDS library functions correctly, but they clutter output. This module
filters those specific warnings at import time.

Usage: import this module at the top of any script that uses wrds, before
the wrds import itself:

    from axiom_fund import _warnings  # noqa: F401
    import wrds
"""

from __future__ import annotations

import warnings

# Suppress SyntaxWarnings originating from the wrds package source.
# These are harmless library-internal warnings about escape sequences.
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"wrds\..*",
)

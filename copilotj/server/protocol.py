# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Frontend <-> server protocol versioning (issue #68).

The protocol version uses ``MAJOR.MINOR`` semantics, declared as two separate
integers (so the bump intent is explicit) and concatenated into the exposed
``API_VERSION`` string:

- bump ``API_VERSION_MAJOR`` on any breaking protocol change (route shapes,
  NDJSON event contracts, required field removal/rename, etc.);
- bump ``API_VERSION_MINOR`` on backward-compatible additions (new optional
  fields/endpoints).

The combined ``API_VERSION`` string MUST be kept in sync with
``web/src/apis/version.ts`` ``API_VERSION``.

Examples:
    >>> API_VERSION == f"{API_VERSION_MAJOR}.{API_VERSION_MINOR}"
    True
    >>> API_VERSION
    '1.0'
"""

__all__ = ["API_VERSION", "API_VERSION_MAJOR", "API_VERSION_MINOR"]

API_VERSION_MAJOR = 1
API_VERSION_MINOR = 0
API_VERSION = f"{API_VERSION_MAJOR}.{API_VERSION_MINOR}"

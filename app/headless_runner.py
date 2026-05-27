"""Compatibility entrypoint for the Raspberry Pi runtime service.

Use ``python -m app.runtime.detector_service`` for new deployments.
"""

from app.runtime.detector_service import main


if __name__ == "__main__":
    main()

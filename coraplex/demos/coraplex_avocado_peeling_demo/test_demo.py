#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the avocado peeling demo without a viewer and exit non-zero with a traceback on
    failure.

    The avocado that comes with CRAM is used, unless ``AVOCADO_DIRECTORY`` names
    another.
    """
    try:
        import demo

        demo.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

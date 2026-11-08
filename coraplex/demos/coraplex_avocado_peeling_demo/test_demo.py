#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the avocado peeling demo without a viewer and exit non-zero with a traceback on
    failure.

    The avocado model's folder is read from ``AVOCADO_DIRECTORY``.
    """
    try:
        import demo

        demo.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

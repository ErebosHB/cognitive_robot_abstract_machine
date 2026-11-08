#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the PR2 avocado peeling demo without a viewer and exit non-zero with a traceback
    on failure.

    The avocado model's folder is read from ``AVOCADO_DIRECTORY``; the PR2's meshes come
    from ``iai_pr2_description`` in a sourced ROS 2 workspace.
    """
    try:
        import pr2_demo

        pr2_demo.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

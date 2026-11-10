#!/usr/bin/env python
import traceback


def main() -> None:
    """
    Run the PR2 whole-avocado peeling demo without a viewer and exit non-zero with a
    traceback on failure.

    The avocado that comes with CRAM is used, unless ``AVOCADO_DIRECTORY`` names
    another; the PR2's meshes come from ``iai_pr2_description`` in a sourced ROS 2
    workspace.
    """
    try:
        import pr2_whole_avocado_demo

        pr2_whole_avocado_demo.main(headless=True)
    except Exception:
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

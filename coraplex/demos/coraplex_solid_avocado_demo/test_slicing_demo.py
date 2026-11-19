#!/usr/bin/env python
"""
Runs the avocado slicing demo headless, as CI does; a failure exits non-zero with its
traceback.
"""

import slicing_demo

if __name__ == "__main__":
    slicing_demo.main(headless=True)

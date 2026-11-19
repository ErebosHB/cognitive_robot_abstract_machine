#!/usr/bin/env python
"""
Runs the avocado scooping demo headless, as CI does; a failure exits non-zero with its
traceback.
"""

import scooping_demo

if __name__ == "__main__":
    scooping_demo.main(headless=True)

#!/usr/bin/env python
"""
Runs the avocado skin peeling demo headless, as CI does; a failure exits non-zero with its
traceback.
"""

import skin_peeling_demo

if __name__ == "__main__":
    skin_peeling_demo.main(headless=True)

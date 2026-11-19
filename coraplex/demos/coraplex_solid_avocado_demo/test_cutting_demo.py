#!/usr/bin/env python
"""
Runs the avocado cutting demo headless, as CI does; a failure exits non-zero with its
traceback.
"""

import cutting_demo

if __name__ == "__main__":
    cutting_demo.main(headless=True)

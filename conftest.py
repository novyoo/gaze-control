# conftest.py
import sys
import os

# Make sure pytest can always find the src package
sys.path.insert(0, os.path.dirname(__file__))
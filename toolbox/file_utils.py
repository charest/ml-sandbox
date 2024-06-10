from pathlib import Path

###############################################################################

def dirname(f):
    """Return the directory path of a file"""
    return Path(f).resolve().parent


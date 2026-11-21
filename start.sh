#!/bin/sh
# OversprayLab launcher for Linux and macOS (start.bat on Windows): the first launch creates a
# virtual environment and installs the dependencies, then the application opens in the browser.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10 or later is required (python3 not found)."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Python 3.10 or later is required (found $(python3 --version 2>&1))."
    exit 1
fi

# a virtual environment left without pip by a failed creation is made again
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    rm -rf .venv
    if ! python3 -m venv .venv; then
        rm -rf .venv
        echo "Could not create the virtual environment."
        echo "On Debian/Ubuntu, install the venv module first: sudo apt install python3-venv"
        exit 1
    fi
fi

# installs or completes the dependencies (quick when they are already there)
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt || exit 1
# --open: the application opens the browser itself, on the port it actually uses
exec .venv/bin/python app.py --open

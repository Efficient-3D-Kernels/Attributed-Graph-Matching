import os
import sys

# Ensure repository root and website directory are in the Python search path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_THIS_DIR)
WEBSITE_DIR = os.path.join(ROOT_DIR, "website")

for p in [ROOT_DIR, WEBSITE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from website.app import app

# Vercel serverless function entrypoint
# app is the WSGI application instance

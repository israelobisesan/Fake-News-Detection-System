"""
Entry point for the fake news detection web application.

Start the server with::

    python run.py

The trained model must already exist. If it does not, the application still
starts and says so clearly on the home page and at /health, rather than
crashing. Create the artifacts first with::

    python training/prepare_nltk.py     # one-time NLTK resources
    python training/train.py            # trains and saves the model
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure `import app` works no matter which directory run.py is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402

app = create_app()


def main() -> None:
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    # Debug mode is handy while developing but must be off in production: it
    # enables the interactive debugger and reloader.
    debug = os.environ.get("FLASK_DEBUG", "1" if os.environ.get("FLASK_ENV") == "development" else "0") == "1"

    print("=" * 60)
    print("Fake News Detection System")
    print("=" * 60)
    print(f"  URL      : http://{host}:{port}")
    print(f"  Debug    : {debug}")
    print("  Stop     : press Ctrl + C")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
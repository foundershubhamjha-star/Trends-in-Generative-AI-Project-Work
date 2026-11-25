"""Start the Dubbing Studio web interface:  python app.py"""
import logging

from dubstudio.ui import launch

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    launch()

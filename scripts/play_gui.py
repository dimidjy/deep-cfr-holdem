import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui import PokerGUI


NUM_PLAYERS = 2


def main():
    gui = PokerGUI(num_players=NUM_PLAYERS)
    gui.run()


if __name__ == "__main__":
    main()

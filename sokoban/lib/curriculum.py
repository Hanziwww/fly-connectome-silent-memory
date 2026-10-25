"""Four-level game interface; teacher functions are used only in training."""
from game import canonical, distances, encode, legal, optimal, restore, solved, state_from, step

def static_dead(level, state):
    """A sufficient corner-deadlock test, never a timeout-as-deadlock claim."""
    walls, goals = level["walls"], set(level["goals"])
    for box in state[1]:
        if box not in goals and (box-10 in walls or box+10 in walls) and (box-1 in walls or box+1 in walls):
            return True
    return False

"""State and action definitions shared by CapPeeler and its tests."""
from enum import IntEnum


class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    FIRE = 4
    FACE_ENEMY = 5


def encode_state(distance, bearing, own_energy, enemy_energy):
    """Distance: <150/<400/far; energy: <30/<70/high.

    Tank Royale positive bearings turn left: front/right/back/left = 0/1/2/3.
    """
    distance_bin = 0 if distance < 150 else 1 if distance < 400 else 2
    bearing = (bearing + 180) % 360 - 180
    bearing_bin = 0 if -45 <= bearing < 45 else (
        3 if 45 <= bearing < 135 else 1 if -135 <= bearing < -45 else 2)
    def energy_bin(energy):
        return 0 if energy < 30 else 1 if energy < 70 else 2
    return distance_bin, bearing_bin, energy_bin(own_energy), energy_bin(enemy_energy)

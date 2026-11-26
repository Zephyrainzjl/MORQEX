"""Unified MORQEX dispatcher; retain the existing geometry CLI unchanged."""
import sys


def main():
    if sys.argv[1:2] == ['top-damage']:
        from .top_damage.cli import main as top_damage
        return top_damage(sys.argv[2:])
    from .microstructure.cli import main as geometry
    return geometry()

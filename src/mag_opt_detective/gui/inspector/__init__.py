"""Sections of the right inspector (settings of the plot on screen).

Each module exposes ``install(window)``: it builds its section, adds it with
``window.add_inspector_section``, wires it to ``window.controller`` and binds its own settings
keys with ``window.persistence``.
"""

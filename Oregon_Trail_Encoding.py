from typing import List

import numpy as np

from game_files.Oregon_Trail_Classes import OregonTrailState


def encode_observation(state: OregonTrailState):
    """
    Flatten OregonTrailState into a numeric vector for the RL game_files.
    You can tweak normalization and representation as you like.
    """
    features: List[float] = []

    # Party info (pad/truncate to fixed size, e.g., 5 members)
    MAX_PARTY = 5
    for i in range(MAX_PARTY):
        if i < len(state.party):
            member = state.party[i]
            features.append(1.0 if member.alive else 0.0)
            # health between 0 and 1
            features.append(float(member.health))
        else:
            # pad for missing members
            features.append(0.0)  # alive
            features.append(0.0)  # health

    inv = state.inventory
    # Normalize by some "reasonable" max values to keep things ~0–1
    features.extend([
        min(inv.food / 1000.0, 1.0),
        min(inv.clothes / 20.0, 1.0),
        min(inv.ammo / 1000.0, 1.0),
        min(inv.oxen / 20.0, 1.0),
        min(inv.wagon_wheels / 5.0, 1.0),
        min(inv.wagon_axles / 5.0, 1.0),
        min(inv.wagon_tongues / 5.0, 1.0),
        min(inv.money / 1000.0, 1.0),
    ])

    trail = state.trail
    features.extend([
        min(trail.segment_index / 20.0, 1.0),
        min(trail.miles_into_segment / 200.0, 1.0),
        min(trail.miles_total / 2000.0, 1.0),
    ])

    env = state.env
    features.extend([
        env.day / 31.0,
        env.month / 12.0,
        float(env.weather.value),
        float(env.terrain.value),
        env.temperature / 50.0,   # roughly normalize
        env.river_depth / 10.0,
    ])

    # Enums as numeric codes (you can one-hot these later if you want)
    features.append(float(state.pace.value))
    features.append(float(state.rations.value))
    features.append(float(state.location_type.value))

    flags = state.event_flags
    features.extend([
        float(flags.in_hunting_minigame),
        float(flags.in_river_decision),
        float(flags.in_store_menu),
        float(flags.last_event_type),
        float(state.game_over),
        float(state.win),
    ])

    return np.array(features, dtype=np.float32)

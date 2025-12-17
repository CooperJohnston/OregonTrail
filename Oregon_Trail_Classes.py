from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class LocationType(Enum):
    TRAIL = 0
    FORT = 1
    RIVER = 2
    LANDMARK = 3
    TOWN = 4


class Pace(Enum):
    SLOW = 0
    NORMAL = 1
    GRUELING = 2


class Rations(Enum):
    BARE_BONES = 0
    MEAGER = 1
    FILLING = 2


class Weather(Enum):
    CLEAR = 0
    RAIN = 1
    SNOW = 2
    HOT = 3
    COLD = 4


class Terrain(Enum):
    PLAINS = 0
    HILLS = 1
    MOUNTAINS = 2
    DESERT = 3
    FOREST = 4


class Action(Enum):
    """Global action space. Some actions may be illegal in certain states."""
    CONTINUE_ON_TRAIL = 0
    REST = 1
    HUNT = 2
    CHANGE_PACE_SLOW = 3
    CHANGE_PACE_NORMAL = 4
    CHANGE_PACE_GRUELING = 5
    CHANGE_RATIONS_BARE_BONES = 6
    CHANGE_RATIONS_MEAGER = 7
    CHANGE_RATIONS_FILLING = 8
    # River-specific actions (only valid when at river)
    FORD_RIVER = 9
    CAULK_AND_FLOAT = 10
    TAKE_FERRY = 11
    WAIT_AT_RIVER = 12
    BUY_FOOD = 13
    BUY_AMMO = 14
    BUY_CLOTHES = 15
    BUY_OXEN = 16
    BUY_WHEEL = 17
    BUY_AXLE = 18
    BUY_TONGUE = 19

@dataclass
class PartyMember:
    name: str
    alive: bool = True
    health: float = 1.0  # 1.0 = perfect, 0.0 = dead (you can discretize later)


@dataclass
class Inventory:
    food: int = 700
    clothes: int = 10
    ammo: int = 100
    oxen: int = 4
    wagon_wheels: int = 2
    wagon_axles: int = 2
    wagon_tongues: int = 2
    money: int = 500


@dataclass
class TrailPosition:
    segment_index: int = 0        # which leg of the trail
    miles_into_segment: int = 0
    miles_total: int = 0


@dataclass
class EnvironmentState:
    day: int = 1
    month: int = 3                # 3 = March, etc.
    weather: Weather = Weather.CLEAR
    terrain: Terrain = Terrain.PLAINS
    temperature: float = 15.0     # Celsius-ish
    river_depth: float = 0.0      # feet, relevant at river


@dataclass
class EventFlags:
    in_hunting_minigame: bool = False
    in_river_decision: bool = False
    in_store_menu: bool = False
    last_event_type: int = 0      # encode events however you want


@dataclass
class OregonTrailState:
    party: List[PartyMember]
    inventory: Inventory
    trail: TrailPosition
    env: EnvironmentState
    pace: Pace
    rations: Rations
    location_type: LocationType
    event_flags: [EventFlags]
    game_over: bool = False
    win: bool = False

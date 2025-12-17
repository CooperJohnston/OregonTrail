import random
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

import numpy as np

from game_files.Oregon_Trail_Classes import OregonTrailState, Action, PartyMember, Inventory, TrailPosition, \
    EnvironmentState, Weather, Terrain, LocationType, Rations, Pace, EventFlags
from game_files.Oregon_Trail_Encoding import encode_observation


class OregonTrailEnv:
    """
    Minimal RL-friendly environment for an Oregon Trail–like game.
    API is Gym-ish:

        env = OregonTrailEnv()
        obs = env.reset()
        obs, reward, done, info = env.step(action)

    `action` should be an integer corresponding to Action.value.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.state: Optional[OregonTrailState] = None
        self._will_travel_today = True

        self.action_space_n = len(Action)
        self.action_counts = defaultdict(int)
        # Simple config tunes
        self.total_miles_to_oregon = 2000
        self.route_milestones = [
            (0, LocationType.TRAIL, "Independence"),
            (100, LocationType.RIVER, "Kansas River"),
            (300, LocationType.FORT, "Fort Kearny"),
            (600, LocationType.RIVER, "Platte River"),
            (800, LocationType.FORT, "Fort Laramie"),
            (950, LocationType.LANDMARK, "Independence Rock"),
            (1200, LocationType.RIVER, "Green River"),
            (1500, LocationType.FORT, "Fort Boise"),
            (1800, LocationType.RIVER, "Columbia River"),
            (2000, LocationType.TOWN, "Oregon City"),
        ]
        self.store_bundles = {
            Action.BUY_FOOD: (50, 5),  # +50 food for $5
            Action.BUY_AMMO: (20, 4),  # +20 ammo for $4
            Action.BUY_CLOTHES: (1, 8),  # +1 clothes for $8
            Action.BUY_OXEN: (1, 40),  # +1 ox for $40
            Action.BUY_WHEEL: (1, 15),  # +1 wheel for $15
            Action.BUY_AXLE: (1, 15),
            Action.BUY_TONGUE: (1, 15),
        }
        self.hunt_ammo_cost = 10
        self.ferry_cost = 5
        self.prev_miles_total = 0
        self.prev_alive_count = 0
        self.action_counts = defaultdict(int)  # overall executed actions
        self.action_counts_by_terrain = defaultdict(lambda: defaultdict(int))
        self.action_counts_by_location = defaultdict(lambda: defaultdict(int))

    # ---------- Core API ----------
    def _store_price_multiplier(self) -> float:
        s = self.state;
        assert s is not None
        progress = min(max(s.trail.miles_total / self.total_miles_to_oregon, 0.0), 1.0)
        return 1.0 + 1.5 * progress  # up to 2.5x at the end
    def reset(self) -> np.ndarray:
        """Start a new game and return initial observation."""
        party = [
            PartyMember(name="You"),
            PartyMember(name="Alice"),
            PartyMember(name="Bob"),
            PartyMember(name="Charlie"),
            PartyMember(name="Dana"),
        ]

        inventory = Inventory(
            food=500,
            clothes=10,
            ammo=100,
            oxen=4,
            wagon_wheels=2,
            wagon_axles=2,
            wagon_tongues=2,
            money=500,
        )

        trail = TrailPosition(
            segment_index=0,
            miles_into_segment=0,
            miles_total=0,
        )

        env = EnvironmentState(
            day=1,
            month=3,  # March
            weather=Weather.CLEAR,
            terrain=Terrain.PLAINS,
            temperature=15.0,
            river_depth=0.0,
        )

        flags = EventFlags()

        self.state = OregonTrailState(
            party=party,
            inventory=inventory,
            trail=trail,
            env=env,
            pace=Pace.NORMAL,
            rations=Rations.MEAGER,
            location_type=LocationType.TRAIL,
            event_flags=flags,
        )
        self.action_counts.clear()
        self.prev_miles_total = self.state.trail.miles_total
        self.prev_alive_count = sum(1 for m in self.state.party if m.alive)
        self.prev_miles_total = self.state.trail.miles_total
        self.prev_alive_count = sum(1 for m in self.state.party if m.alive)
        self.prev_food = self.state.inventory.food
        self.prev_money = self.state.inventory.money

        return encode_observation(self.state)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply an action, advance the game by one "day" (or decision step),
        and return (obs, reward, done, info).
        """
        assert self.state is not None, "Call reset() before step()."
        if self.state.game_over:
            obs = encode_observation(self.state)
            return obs, 0.0, True, {"terminal": True}

        action = Action(action_idx)


        # Optional: enforce legality inside env
        if action_idx not in self.legal_actions():
            # Option A: treat as "REST" (safe fallback)
            action = Action.REST
            illegal_penalty = -2.0
        else:
            illegal_penalty = 0.0
        # --- legality + fallback ---
        action = Action(action_idx)
        if action_idx not in self.legal_actions():
            action = Action.REST
            illegal_penalty = -2.0
        else:
            illegal_penalty = 0.0

        # --- count executed action by context ---
        a = action.value
        terrain = self.state.env.terrain
        loc = self.state.location_type

        self.action_counts[a] += 1
        self.action_counts_by_terrain[terrain][a] += 1
        self.action_counts_by_location[loc][a] += 1

        prev_food = self.prev_food
        prev_money = self.prev_money
        # --- Apply action decision ---
        self._apply_action(action)
        self._will_travel_today = (action == Action.CONTINUE_ON_TRAIL)
        self._advance_time_and_events()
        self.food_delta = self.state.inventory.food - prev_food
        self.money_delta = self.state.inventory.money - prev_money  # negative when spending

        self._update_terminal_status()
        reward = self._compute_reward() + illegal_penalty

        obs = encode_observation(self.state)
        done = self.state.game_over
        info: Dict = {}

        self.prev_miles_total = self.state.trail.miles_total
        self.prev_alive_count = sum(1 for m in self.state.party if m.alive)
        self.prev_food = self.state.inventory.food
        self.prev_money = self.state.inventory.money

        return obs, reward, done, info

    # ---------- Internal Helpers ----------

    def _apply_action(self, action: Action) -> None:
        """Modify state based on chosen action."""
        s = self.state
        assert s is not None

        # Some actions only make sense in some locations;

        if action == Action.CONTINUE_ON_TRAIL:
            # actual movement is handled in _advance_time_and_events
            pass

        elif action == Action.REST:
            # Rest: don't move, but improve health a bit
            for member in s.party:
                if member.alive:
                    member.health = min(member.health + 0.1, 1.0)

        elif action in self.store_bundles:
            if s.location_type not in (LocationType.FORT, LocationType.TOWN):
                s.event_flags.last_event_type = 11  # tried to buy not at store
                return

            qty, base_cost = self.store_bundles[action]
            cost = int(round(base_cost * self._store_price_multiplier()))
            if s.inventory.money < cost:
                s.event_flags.last_event_type = 13  # not enough money
                return

            s.inventory.money -= cost

            if action == Action.BUY_FOOD:
                s.inventory.food += qty
            elif action == Action.BUY_AMMO:
                s.inventory.ammo += qty
            elif action == Action.BUY_CLOTHES:
                s.inventory.clothes += qty
            elif action == Action.BUY_OXEN:
                s.inventory.oxen += qty
            elif action == Action.BUY_WHEEL:
                s.inventory.wagon_wheels += qty
            elif action == Action.BUY_AXLE:
                s.inventory.wagon_axles += qty
            elif action == Action.BUY_TONGUE:
                s.inventory.wagon_tongues += qty


            s.event_flags.last_event_type = 12  # bought somethin

        elif action == Action.HUNT:
            # Hunting takes time (handled in _advance_time_and_events),
            # consumes ammo, and may yield food.
            HUNT_AMMO_COST = 10

            if s.inventory.ammo < HUNT_AMMO_COST:
                # Not enough ammo: no hunt, small "frustration" event
                s.event_flags.last_event_type = 2  # "tried to hunt with no ammo"
                return

            s.inventory.ammo -= HUNT_AMMO_COST

            # Base success probability depends on terrain
            base_success = {
                Terrain.PLAINS: 0.7,
                Terrain.FOREST: 0.8,
                Terrain.HILLS: 0.6,
                Terrain.MOUNTAINS: 0.5,
                Terrain.DESERT: 0.3,
            }[s.env.terrain]

            # Slightly less success in bad weather
            weather_penalty = {
                Weather.CLEAR: 1.0,
                Weather.HOT: 0.9,
                Weather.COLD: 0.9,
                Weather.RAIN: 0.8,
                Weather.SNOW: 0.7,
            }[s.env.weather]

            success_prob = base_success * weather_penalty

            if self.rng.random() < success_prob:
                # Successful hunt: yield some food
                food_min, food_max = 40, 160  # in pounds
                food_gained = self.rng.randint(food_min, food_max)
                s.inventory.food += food_gained
                s.event_flags.last_event_type = 1  # "successful hunt"
            else:
                # Unsuccessful hunt
                s.event_flags.last_event_type = 5  # "unsuccessful hunt"

        elif action == Action.CHANGE_PACE_SLOW:
            s.pace = Pace.SLOW

        elif action == Action.CHANGE_PACE_NORMAL:
            s.pace = Pace.NORMAL

        elif action == Action.CHANGE_PACE_GRUELING:
            s.pace = Pace.GRUELING

        elif action == Action.CHANGE_RATIONS_BARE_BONES:
            s.rations = Rations.BARE_BONES

        elif action == Action.CHANGE_RATIONS_MEAGER:
            s.rations = Rations.MEAGER

        elif action == Action.CHANGE_RATIONS_FILLING:
            s.rations = Rations.FILLING

        # River-specific choices (only meaningful at rivers)
        elif action in [
            Action.FORD_RIVER,
            Action.CAULK_AND_FLOAT,
            Action.TAKE_FERRY,
            Action.WAIT_AT_RIVER,
        ]:
            # TODO: Implement river logic with probabilities of losing oxen, items, people, etc.
            # For now, we just mark an event and treat them as "wasted time" choices.
            if s.location_type != LocationType.RIVER:
                s.event_flags.last_event_type = 6  # "river action not at river"
                return

            depth = s.env.river_depth

            if action == Action.WAIT_AT_RIVER:
                # Wait a day, river *might* get shallower
                s.env.river_depth = max(1.0, depth - self.rng.uniform(0.1, 0.7))
                s.event_flags.last_event_type = 7  # "waited at river"
                return

            if action == Action.TAKE_FERRY:
                # Pay some money, safe crossing
                ferry_cost = 5  # adjust
                if s.inventory.money >= ferry_cost:
                    s.inventory.money -= ferry_cost
                # Mark that we've effectively "crossed":
                # simplest: give a small artificial distance jump
                s.trail.miles_total += 30
                s.event_flags.last_event_type = 8  # "took ferry"
                return

            # FORD or CAULK: risk depends heavily on depth
            # Rough risk model:
            # - Shallow (<3 ft): pretty safe
            # - Medium (3–5 ft): risky
            # - Deep (>5 ft): very risky

            if depth < 3.0:
                base_risk = 0.05
            elif depth < 5.0:
                base_risk = 0.2
            else:
                base_risk = 0.4

            if action == Action.FORD_RIVER:
                risk = base_risk * 1.2
            else:  # CAULK_AND_FLOAT
                risk = base_risk * 0.8

            if self.rng.random() < risk:
                # Disaster: lose some supplies and possible party member
                s.event_flags.last_event_type = 9  # "river disaster"
                # Lose random fraction of food and ammo
                s.inventory.food = int(s.inventory.food * self.rng.uniform(0.3, 0.7))
                s.inventory.ammo = int(s.inventory.ammo * self.rng.uniform(0.3, 0.7))

                # Chance an ox dies
                if s.inventory.oxen > 0 and self.rng.random() < 0.5:
                    s.inventory.oxen -= 1

                # Chance a party member dies
                alive_members = [m for m in s.party if m.alive]
                if alive_members and self.rng.random() < 0.3:
                    victim = self.rng.choice(alive_members)
                    victim.alive = False
                    victim.health = 0.0
            else:
                # Safe crossing
                s.event_flags.last_event_type = 10  # "river success"

            # Either way, consider us as having crossed; small forward jump:
            s.trail.miles_total += 30

    def _apply_no_spare_consequence(self) -> None:
        s = self.state;
        assert s is not None
        s.event_flags.last_event_type = 24  # no spare

        # Lose a day of progress (simulate delay)
        # easiest RL-friendly: cancel travel tomorrow by rolling back some miles
        rollback = self.rng.randint(5, 15)
        s.trail.miles_total = max(0, s.trail.miles_total - rollback)

        # Maybe lose an ox trying to drag the wagon
        if s.inventory.oxen > 0 and self.rng.random() < 0.30:
            s.inventory.oxen -= 1
            s.event_flags.last_event_type = 25  # lost ox

        # Minor health hit to all alive
        for m in s.party:
            if m.alive:
                m.health = max(0.0, m.health - 0.05)
                if m.health <= 0:
                    m.alive = False

    def _advance_time_and_events(self) -> None:
        """Advance one day, update miles, food, health, and apply random events."""
        s = self.state
        assert s is not None

        # Advance date (very simplified)
        s.env.day += 1
        if s.env.day > 30:
            s.env.day = 1
            s.env.month += 1

        # Daily movement based on pace
        if self._will_travel_today:
            base_miles = {
                Pace.SLOW: 10,
                Pace.NORMAL: 15,
                Pace.GRUELING: 20,
            }[s.pace]

            weather_factor = {
                Weather.CLEAR: 1.0,
                Weather.RAIN: 0.8,
                Weather.SNOW: 0.5,
                Weather.HOT: 0.9,
                Weather.COLD: 0.9,
            }[s.env.weather]

            miles_today = int(base_miles * weather_factor)
            if s.inventory.oxen <= 0:
                miles_today = 0

            s.trail.miles_into_segment += miles_today
            s.trail.miles_total += miles_today

            # Update location based on new total miles
        self._update_location_by_miles()

        # Food consumption per day based on rations and living party members
        living_members = sum(1 for m in s.party if m.alive)
        ration_factor = {
            Rations.BARE_BONES: 1,
            Rations.MEAGER: 2,
            Rations.FILLING: 3,
        }[s.rations]

        food_consumed = living_members * ration_factor
        s.inventory.food -= food_consumed
        if s.inventory.food < 0:
            s.inventory.food = 0

        # --- Wagon breakdown event ---
        if self.rng.random() < self._breakdown_probability():
            part = self.rng.choice(["wheel", "axle", "tongue"])
            s.event_flags.last_event_type = 20  # breakdown

            if part == "wheel":
                if s.inventory.wagon_wheels > 0:
                    s.inventory.wagon_wheels -= 1
                    s.event_flags.last_event_type = 21  # used wheel
                else:
                    self._apply_no_spare_consequence()
            elif part == "axle":
                if s.inventory.wagon_axles > 0:
                    s.inventory.wagon_axles -= 1
                    s.event_flags.last_event_type = 22
                else:
                    self._apply_no_spare_consequence()
            else:
                if s.inventory.wagon_tongues > 0:
                    s.inventory.wagon_tongues -= 1
                    s.event_flags.last_event_type = 23
                else:
                    self._apply_no_spare_consequence()

        # Health effects from rations and pace (very rough placeholder game_files)
        # Health effects: event/condition-based (no automatic daily decay)
        for m in s.party:
            if not m.alive:
                continue

            health_change = 0.0

            # Good conditions -> small recovery
            if s.rations == Rations.FILLING and s.pace != Pace.GRUELING and s.inventory.food > 0:
                health_change += 0.01
            if not self._will_travel_today:
                health_change += 0.02
            # Bad conditions -> gradual decline
            if s.rations == Rations.BARE_BONES:
                health_change -= 0.03
            if s.pace == Pace.GRUELING:
                health_change -= 0.02
            if s.inventory.food == 0:
                health_change -= 0.15

            # Disease probability depends on conditions (risk-based)
            disease_prob = 0.01
            if s.pace == Pace.GRUELING:
                disease_prob += 0.02
            if s.rations == Rations.BARE_BONES:
                disease_prob += 0.03
            if s.env.weather in (Weather.COLD, Weather.SNOW):
                disease_prob += 0.02
            disease_prob = min(disease_prob, 0.04)
            if self.rng.random() < disease_prob:
                health_change -= self.rng.uniform(0.05, 0.15)
                s.event_flags.last_event_type = 4  # "disease"

            # Apply + clamp
            m.health = max(0.0, min(1.0, m.health + health_change))
            if m.health <= 0.0:
                m.alive = False

        # Simple weather randomization (placeholder)
        if self.rng.random() < 0.1:
            s.env.weather = self.rng.choice(list(Weather))

    def _compute_reward(self) -> float:
        s = self.state
        assert s is not None

        miles_now = s.trail.miles_total
        delta_miles = max(0, miles_now - self.prev_miles_total)

        alive_now = sum(1 for m in s.party if m.alive)
        deaths_this_step = max(0, self.prev_alive_count - alive_now)
        reward = 0.0

        # progress
        reward += 0.4 * delta_miles

        # time
        reward -= 0.01

        # deaths
        #reward -= 250.0 * deaths_this_step

        # money spent (delta_money is negative when you spend)

        # food change (reward gaining food a bit, penalize losing food more)
        delta_money = self.state.inventory.money - self.prev_money
        reward += 0.02 * (delta_money)
        if s.inventory.food == 0:
            reward -= 2500.0
        #elif s.inventory.food < 80:
            #reward -= 1.0

        # terminal
        if s.game_over:
            reward += (100000.0 if s.win else -1000.0)

        return float(reward)

    def _update_terminal_status(self) -> None:
        """Check whether the game is over (win or lose)."""
        s = self.state
        assert s is not None

        # Lose: no living party members
        any_alive = any(m.alive for m in s.party)
        if not any_alive:
            s.game_over = True
            s.win = False
            return

        # Lose: no oxen
        if s.inventory.oxen <= 0:
            s.game_over = True
            s.win = False
            return



        # Lose: too late in the year (e.g., past October)
        if s.env.month > 10:
            s.game_over = True
            s.win = False
            return

        # Win: reached Oregon
        if s.trail.miles_total >= self.total_miles_to_oregon:
            s.game_over = True
            s.win = True
            return

    def _update_location_by_miles(self) -> None:
        """
        Update location_type as a short encounter window around milestones
        (river/fort/landmark), instead of staying 'at the river' for hundreds of miles.
        """
        s = self.state
        assert s is not None

        miles = s.trail.miles_total
        if miles < 400:
            s.env.terrain = Terrain.PLAINS
        elif miles < 900:
            s.env.terrain = Terrain.HILLS
        elif miles < 1300:
            s.env.terrain = Terrain.MOUNTAINS
        else:
            s.env.terrain = Terrain.PLAINS
        # Default: you're on the trail
        s.location_type = LocationType.TRAIL
        s.event_flags.in_river_decision = False
        s.env.river_depth = 0.0

        # If you're within this many miles after a milestone, you're "at" that milestone.
        # Must be larger than your max daily movement so you don't skip it.
        WINDOW = 25

        for mile_marker, loc_type, name in reversed(self.route_milestones):
            if mile_marker <= miles < mile_marker + WINDOW:
                s.location_type = loc_type

                if loc_type == LocationType.RIVER:
                    base_depth = {
                        3: 3.0,  # March
                        4: 4.0,  # April
                        5: 5.0,  # May
                        6: 4.0,
                        7: 3.0,
                        8: 2.5,
                        9: 2.0,
                        10: 1.5,
                    }.get(s.env.month, 3.0)

                    s.env.river_depth = max(1.0, self.rng.gauss(base_depth, 0.7))
                    s.event_flags.in_river_decision = True

                break


    def _is_legal(self, action_idx: int) -> bool:
        return action_idx in set(self.legal_actions())

    def _breakdown_probability(self) -> float:
        s = self.state;
        assert s is not None
        p = 0.001  # base 1% per day

        if s.pace == Pace.GRUELING:
            p += 0.003
        elif s.pace == Pace.NORMAL:
            p += 0.001

        if s.env.terrain in (Terrain.MOUNTAINS, Terrain.HILLS):
            p += 0.002

        if s.env.weather in (Weather.RAIN, Weather.SNOW):
            p += 0.001

        return min(p, 0.05)  # cap at 5%

    def legal_actions(self) -> List[int]:
        """
        Return a list of action indices (ints) that are legal/meaningful
        in the current state.
        """
        s = self.state
        if s is None:
            # Before reset, nothing is legal
            return []

        if s.game_over:
            # Episode is done; you can decide to return [] or allow a "no-op"
            return []

        legal: set[int] = set()

        # --- Actions that are always reasonable ---

        # Rest is always allowed
        legal.add(Action.REST.value)

        # You can always change pace and rations
        legal.add(Action.CHANGE_PACE_SLOW.value)
        legal.add(Action.CHANGE_PACE_NORMAL.value)
        legal.add(Action.CHANGE_PACE_GRUELING.value)

        legal.add(Action.CHANGE_RATIONS_BARE_BONES.value)
        legal.add(Action.CHANGE_RATIONS_MEAGER.value)
        legal.add(Action.CHANGE_RATIONS_FILLING.value)

        # --- Location-dependent actions ---

        if s.location_type in (
            LocationType.TRAIL,
            LocationType.FORT,
            LocationType.LANDMARK,
            LocationType.TOWN,
        ):
            if s.location_type in (LocationType.FORT, LocationType.TOWN):
                # allow purchases only if you can afford at least the cheapest thing
                legal.add(Action.BUY_FOOD.value)
                legal.add(Action.BUY_AMMO.value)
                legal.add(Action.BUY_CLOTHES.value)
                legal.add(Action.BUY_OXEN.value)
                legal.add(Action.BUY_WHEEL.value)
                legal.add(Action.BUY_AXLE.value)
                legal.add(Action.BUY_TONGUE.value)
            # General overland travel locations:
            # You can continue on the trail
            legal.add(Action.CONTINUE_ON_TRAIL.value)

            # Hunting allowed if we have enough ammo
            if s.inventory.ammo >= self.hunt_ammo_cost:
                legal.add(Action.HUNT.value)

        if s.location_type == LocationType.RIVER:
            #legal.add(Action.CONTINUE_ON_TRAIL.value)
            # At a river, we allow the river-specific choices:
            legal.add(Action.FORD_RIVER.value)
            legal.add(Action.CAULK_AND_FLOAT.value)
            legal.add(Action.WAIT_AT_RIVER.value)

            # Ferry only if we have enough money
            if s.inventory.money >= self.ferry_cost:
                legal.add(Action.TAKE_FERRY.value)

            # You can also still rest/hunt/adjust pace/rations at the riverbank
            # Hunting only if enough ammo:
            if s.inventory.ammo >= self.hunt_ammo_cost:
                legal.add(Action.HUNT.value)

        # You might choose to disallow CONTINUE_ON_TRAIL while in a "river decision"
        # state, but with our current model we handle river crossing via those actions.

        # Return as a sorted list for determinism
        return sorted(legal)

"""
Site feasibility — what fits on this plot *before* the owner presses Generate.

The Design form asks for the plot first, then the family / BHK, then the site amenities.
This module answers, for the inputs given so far, which later choices can still be honoured,
so the form can switch off what will not fit (with the reason and what would unlock it)
instead of letting the owner ask for a car bay that the generator then silently drops or
squeezes in by shrinking every room.

It runs the generator's own SitePlanner (setbacks, parking/pool/garden reservations, amenity
placement) — the answer matches what Generate will draw — and adds one owner-level rule:

  An amenity that takes space from the house is not offered when the house left over would
  fall below COMFORT × the comfortable footprint of the chosen BHK (Standard room sizes,
  + walls/circulation) — e.g. a car bay on a 3-cent 2BHK. Between COMFORT and TIGHT it is
  offered with a warning ("the house gets 6 m² smaller").

Pure Python, no Streamlit; every call is a handful of planner runs (milliseconds).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from floorplan_generator.site_planner import SitePlanner
from floorplan_generator.space_program import build_program

COMFORT = 0.80    # house keeps ≥ 80 % of a comfortable footprint, or the amenity is not offered
TIGHT = 0.92      # below this the amenity is offered with a "house gets smaller" warning
LOSS = 0.04       # an amenity costing the house < 4 % never counts as squeezing it
SQM_TO_SQFT = 10.7639

# Order = the planner's own priority: parking claims space before pool and garden
AMENITIES = ["car_parking", "bike_parking", "swimming_pool", "garden",
             "water_tank", "generator_room", "security_cabin"]
CATEGORY = {"car_parking": "PARKING", "bike_parking": "BIKE_PARKING", "swimming_pool": "POOL",
            "garden": "GARDEN", "water_tank": "WATER_TANK", "generator_room": "GENERATOR",
            "security_cabin": "SECURITY"}
LABEL = {"car_parking": "Car parking", "bike_parking": "Two-wheeler parking",
         "swimming_pool": "Swimming pool", "garden": "Garden", "water_tank": "Water sump",
         "generator_room": "Generator", "security_cabin": "Security cabin"}
POOL_SIZES = ["Small (3m × 6m)", "Medium (4m × 8m)", "Large (5m × 10m)"]
GARDEN_PCTS = [5, 10, 15, 20, 25]
BIKE_SLOTS = list(range(2, 21, 2))
MAX_CARS = 8
STILT_M2_PER_CAR = 25.0


@dataclass
class Fit:
    """Whether one choice can be honoured. status: 'ok' | 'tight' | 'no'."""
    key: str
    status: str
    reason: str = ""
    unlock: str = ""                      # what would make a 'no' possible
    house_m2: float = 0.0                 # house footprint with this choice
    loss_m2: float = 0.0                  # house area it costs
    max_value: Optional[int] = None       # car / bike slots, garden %
    amount_m2: float = 0.0                # garden / pool area actually laid out
    choices: List[str] = field(default_factory=list)   # pool sizes that fit
    hint: Dict = field(default_factory=dict)            # extra config to pass to the engine


@dataclass
class SiteFit:
    bhk: str
    comfy_m2: float                       # comfortable house footprint for this BHK
    house_m2: float                       # house footprint with the accepted amenities
    min_m2: float                         # below this the generator drops a bedroom
    bhk_fit: Dict[str, Fit]
    amenities: Dict[str, Fit]
    lifestyle_dropped: List[str]          # optional rooms the generator will have to leave out

    def summary(self) -> str:
        """One plain sentence for the top of the amenities section."""
        no = [LABEL[k] for k in AMENITIES if self.amenities[k].status == "no"]
        ok = [LABEL[k] for k in AMENITIES if self.amenities[k].status != "no"]
        if not no:
            return "Everything below fits on your plot."
        lead = ", ".join(no[:-1]) + (" and " if len(no) > 1 else "") + no[-1]
        verb = "does" if len(no) == 1 else "do"
        return f"{lead} {verb} not fit with a {self.bhk} on this plot" + (
            f" — {', '.join(ok[:2]).lower()}{' and more' if len(ok) > 2 else ''} can be added." if ok else ".")


class _Ctx:
    """Inputs shared by all simulations of one form state."""

    def __init__(self, plot_w, plot_h, facing, bhk, family, building_type, num_floors,
                 quality, vastu, has_lift, lift_capacity, stair_type, lift_position="inside"):
        self.W, self.H = float(plot_w), float(plot_h)
        self.facing = (facing or "N").upper()[0]
        self.bhk = bhk
        self.family = family
        self.btype = building_type or "Independent House"
        self.floors = max(1, int(num_floors or 1))
        self.quality = quality or "Standard"
        self.vastu = bool(vastu)
        self.has_lift, self.lift_capacity, self.stair_type = has_lift, lift_capacity, stair_type
        self.lift_position = lift_position or "inside"
        self.members = 4
        if family is not None:
            self.members = max(1, getattr(family, "adults", 2) + getattr(family, "children", 0)
                               + getattr(family, "elderly", 0))
        self.multi_level = (self.floors > 1 or "apartment" in self.btype.lower()
                            or "duplex" in self.btype.lower())

    def program(self, bhk: str, quality: Optional[str] = None, family="same"):
        return build_program(bhk, self.family if family == "same" else family, self.btype, self.floors,
                             quality or self.quality, self.has_lift, self.lift_capacity, self.stair_type,
                             lift_position=self.lift_position)

    def scaled(self, k: float) -> "_Ctx":
        c = _Ctx.__new__(_Ctx)
        c.__dict__.update(self.__dict__)
        c.W, c.H = self.W * math.sqrt(k), self.H * math.sqrt(k)
        return c


def _min_wd(bhk: str) -> Tuple[float, float]:
    n = int(bhk[0]) if bhk and bhk[0].isdigit() else 2
    return {1: 5.6, 2: 6.6, 3: 7.6, 4: 8.4}.get(n, 6.6), 6.8


def _nominal_door(site, planner: SitePlanner) -> Tuple[float, float]:
    """Main-door position the solver is likely to pick: best entrance pada clear of a car bay."""
    from floorplan_generator.vastu_engine import pada_for_point
    fx, fy, fw, fd = site.footprint
    ex = site.door_exclusion
    best = None
    x = fx + 0.45
    while x <= fx + fw - 1.45 + 1e-6:
        if not ex or x >= ex[1] or x + 1.0 <= ex[0]:
            sc = pada_for_point(x + 0.5, fy + fd, site.footprint, planner.facing).score if planner.vastu else 0
            key = sc - 0.05 * abs(x + 0.5 - (fx + fw / 2))
            if best is None or key > best[0]:
                best = (key, x)
        x += 0.1
    x0 = best[1] if best else fx + fw / 2 - 0.5
    return x0, x0 + 1.0


def _simulate(ctx: _Ctx, program, comps: Dict, place: bool = True):
    """Run the generator's site planner for one amenity set → (site, placed categories)."""
    c = dict(comps)
    c["_has_lift"] = bool(program.lift)
    planner = SitePlanner(ctx.W, ctx.H, ctx.facing, ctx.vastu, ctx.btype, ctx.floors, c, ctx.members)
    min_w, min_d = _min_wd(program.bhk)
    ext = program.external_stair.footprint_l if program.external_stair else 0.0
    site = planner.plan_footprint(program.target_area * 1.12, program.min_area, min_w, min_d, ext,
                                  lift_out=_lift_out(program))
    placed = set()
    if place:
        d0, d1 = _nominal_door(site, planner)
        planner.place_amenities(site, d0, d1, program.external_stair)
        placed = {it.category for it in site.items}
        if site.car_mode == "stilt":
            placed.add("PARKING")
    return site, placed


def _lift_out(program) -> Optional[Dict]:
    """The engine's lift-tower reservation for a program with the lift outside (else None)."""
    if not getattr(program, "lift_outside", None) or not program.lift:
        return None
    return {"position": program.lift_outside, "w": program.lift.shaft_w, "d": program.lift.shaft_d, "door": 0.9}


def _house(site) -> float:
    return site.footprint[2] * site.footprint[3]


def _note(site, *words) -> str:
    return next((n for n in site.notes if all(w in n for w in words)), "")


def _sqft(m2: float) -> str:
    return f"{m2:.0f} m² ({m2 * SQM_TO_SQFT:.0f} sq ft)"


def _cfg(key: str, value=None) -> Dict:
    if key == "car_parking":
        return {"enabled": True, "slots": int(value or 1)}
    if key == "bike_parking":
        return {"enabled": True, "slots": int(value or 2)}
    if key == "garden":
        return {"enabled": True, "pct": int(value or 10)}
    if key == "swimming_pool":
        return {"enabled": True, "size": value or POOL_SIZES[0]}
    return {"enabled": True}


class _Checker:
    def __init__(self, ctx: _Ctx):
        self.ctx = ctx
        self.prog = ctx.program(ctx.bhk)                      # what the engine will build
        self.std = ctx.program(ctx.bhk, quality="Standard")   # comfort reference
        self.comfy = self.std.target_area * 1.12
        self._cache: Dict = {}

    def _run(self, comps: Dict, std: bool):
        key = (std, tuple(sorted((k, tuple(sorted(v.items()))) for k, v in comps.items())))
        if key not in self._cache:
            self._cache[key] = _simulate(self.ctx, self.std if std else self.prog, comps, place=not std)
        return self._cache[key]

    def judge(self, accepted: Dict, key: str, value=None) -> Fit:
        """Can `key` (with `value`) be added on top of the amenities already `accepted`?"""
        ctx = self.ctx
        comps = dict(accepted)
        comps[key] = _cfg(key, value)
        site, placed = self._run(comps, std=False)
        cat = CATEGORY[key]
        if key == "car_parking" and site.car_mode is None:
            return Fit(key, "no", "the plot is too small for a car bay next to the house")
        if key == "swimming_pool" and _note(site, "Swimming pool skipped"):
            return Fit(key, "no", "no open space on the North / East side is big enough for a pool")
        if cat not in placed:
            why = {"bike_parking": "no open strip 2 m deep is left for bikes",
                   "garden": "no open strip is left for a garden",
                   "water_tank": "no open corner is left for the sump",
                   "generator_room": "no open space in a suitable Vastu zone is left for the generator",
                   "security_cabin": "the front yard is too shallow for a 1.8 m cabin",
                   "car_parking": "a car bay cannot be kept clear of the entrance path",
                   "swimming_pool": "no open space is big enough for a pool"}[key]
            return Fit(key, "no", why)

        got = sum(it.rect[2] * it.rect[3] for it in site.items if it.category == cat)
        stilt = key == "car_parking" and site.car_mode == "stilt"
        if key not in ("car_parking", "swimming_pool", "garden") or stilt:
            reason = "parked on a stilt floor under the house" if stilt else "fits in the open area"
            return Fit(key, "ok", reason, house_m2=_house(site))

        # Space-taking amenity: how much house does it cost (Standard room sizes)?
        without = _house(self._run(dict(accepted), std=True)[0])
        with_ = _house(self._run(comps, std=True)[0])
        loss = max(0.0, without - with_)
        squeezes = loss > LOSS * without
        if squeezes and (with_ < self.std.min_area or with_ < COMFORT * self.comfy):
            if key == "car_parking" and ctx.multi_level:
                return Fit(key, "ok", "an open bay would squeeze the rooms, so the car goes on a stilt "
                                      "floor under the house", house_m2=without, hint={"mode": "stilt"})
            what = {"car_parking": "a car bay", "swimming_pool": "a pool", "garden": "a garden this big"}[key]
            return Fit(key, "no", f"{what} would take {_sqft(loss)} from the house — every room of the "
                                  f"{ctx.bhk} would be smaller than comfortable",
                       house_m2=with_, loss_m2=loss, amount_m2=got)
        if squeezes and with_ < TIGHT * self.comfy:
            return Fit(key, "tight", f"fits, but the house gets {_sqft(loss)} smaller",
                       house_m2=with_, loss_m2=loss, amount_m2=got)
        reason = {"car_parking": "fits in the front yard" if site.car_mode in ("front", "front_parallel")
                  else "fits along the side of the house",
                  "swimming_pool": "fits in the North / East open area",
                  "garden": "fits in the North / East open area"}[key]
        if key == "garden":
            reason = f"about {_sqft(got)} of garden in the open area"
        return Fit(key, "ok", reason, house_m2=with_, loss_m2=loss, amount_m2=got)


def _unlock(ctx: _Ctx, key: str, value, accepted: Dict) -> str:
    """Plain hint for a 'no': smaller BHK, an extra floor, or about how many cents it needs."""
    hints = []
    if accepted and _Checker(ctx).judge({}, key, value).status != "no":
        hints.append("switch off " + " / ".join(LABEL[k].lower() for k in AMENITIES if k in accepted))
    n = int(ctx.bhk[0])
    if n > 1:
        smaller = f"{n - 1}BHK"
        c2 = _Ctx.__new__(_Ctx)
        c2.__dict__.update(ctx.__dict__)
        c2.bhk = smaller
        if _Checker(c2).judge(accepted, key, value).status != "no":
            hints.append(f"choose {smaller}")
    if key == "car_parking" and not ctx.multi_level:
        hints.append("build 2 floors (car under the house)")
    plot_cents = ctx.W * ctx.H / 40.4686
    for step in range(1, 25):                       # same shape, +0.25 cent steps
        cents = plot_cents + 0.25 * step
        k = cents / plot_cents
        if _Checker(ctx.scaled(k)).judge(accepted, key, value).status != "no":
            hints.append(f"a plot of about {math.ceil(cents * 4) / 4:g} cents")
            break
    if not hints:
        return ""
    if hints == [h for h in hints if h.startswith("a plot")]:
        return "Possible with " + hints[0] + "."
    return "Possible if you " + ", or ".join(h if not h.startswith("a plot") else "have " + h for h in hints) + "."


def check_site(plot_w: float, plot_h: float, facing: str = "N", bhk: str = "2BHK",
               family=None, building_type: str = "Independent House", num_floors: int = 1,
               quality: str = "Standard", vastu: bool = True, has_lift: bool = False,
               lift_capacity: str = "4 Person", stair_type: str = "dog_leg",
               components: Optional[Dict] = None, unlock_hints: bool = True,
               lift_position: str = "inside") -> SiteFit:
    """
    Feasibility of every BHK and site amenity for the form state. `components` is the
    design form's additional_components dict (what is switched on now); amenities are judged
    in planner priority order, each on top of the ones before it that are on and fit.
    """
    ctx = _Ctx(plot_w, plot_h, facing, bhk, family, building_type, num_floors, quality, vastu,
               has_lift, lift_capacity, stair_type, lift_position)
    components = components or {}
    chk = _Checker(ctx)
    bhk_fit = _bhk_options(ctx)

    # ── amenities, in planner priority ──────────────────────────────────
    accepted: Dict = {}
    fits: Dict[str, Fit] = {}
    for key in AMENITIES:
        cfg = components.get(key) or {}
        on = bool(cfg.get("enabled"))
        if key == "car_parking":
            want = int(cfg.get("slots") or 1) if on else 1
            fit = chk.judge(accepted, key, want if on else 1)
            if fit.status != "no":
                top = 1
                for s in range(2, MAX_CARS + 1):
                    if chk.judge(accepted, key, s).status == "no":
                        break
                    top = s
                if "stilt" in fit.reason:     # ~25 m² of stilt floor per car incl. its share of the aisle
                    top = min(top, max(1, int(fit.house_m2 // STILT_M2_PER_CAR)))
                fit.max_value = top
                if on and want > top:
                    fit = chk.judge(accepted, key, top)
                    fit.max_value = top
            value = min(want, fit.max_value or want)
        elif key == "bike_parking":
            want = int(cfg.get("slots") or 2) if on else 2
            fit = chk.judge(accepted, key, 2)
            if fit.status != "no":
                top = 2
                for s in BIKE_SLOTS[1:]:
                    if chk.judge(accepted, key, s).status == "no":
                        break
                    top = s
                fit.max_value = top
            value = min(want, fit.max_value or want)
        elif key == "garden":
            want = int(cfg.get("pct") or 10) if on else 5
            fit = chk.judge(accepted, key, 5)
            if fit.status != "no":
                plot = ctx.W * ctx.H
                top = 5
                for p in GARDEN_PCTS[1:]:
                    f = chk.judge(accepted, key, p)
                    if f.status == "no" or f.amount_m2 < 0.85 * plot * p / 100.0:
                        break
                    top = p
                fit.max_value = top
                if on:
                    fit_w = chk.judge(accepted, key, min(want, top))
                    fit_w.max_value = top
                    fit = fit_w
            value = min(want, fit.max_value or want)
        elif key == "swimming_pool":
            sizes = [s for s in POOL_SIZES if chk.judge(accepted, key, s).status != "no"]
            want = cfg.get("size") or POOL_SIZES[0]
            fit = chk.judge(accepted, key, want if on and want in sizes else (sizes[0] if sizes else POOL_SIZES[0]))
            fit.choices = sizes
            value = want if want in sizes else (sizes[0] if sizes else None)
        else:
            fit = chk.judge(accepted, key)
            value = None
        if fit.status == "no" and unlock_hints:
            fit.unlock = _unlock(ctx, key, value if key != "garden" else 5, accepted)
        fits[key] = fit
        if on and fit.status != "no":
            accepted[key] = dict(_cfg(key, value), **fit.hint)

    final, _ = chk._run(accepted, std=False)
    lifestyle = _lifestyle_dropped(ctx, _house(final))
    return SiteFit(bhk=ctx.bhk, comfy_m2=chk.comfy, house_m2=_house(final), min_m2=chk.prog.min_area,
                   bhk_fit=bhk_fit, amenities=fits, lifestyle_dropped=lifestyle)


def bhk_options(plot_w: float, plot_h: float, facing: str = "N", family=None,
                building_type: str = "Independent House", num_floors: int = 1, vastu: bool = True,
                has_lift: bool = False, lift_capacity: str = "4 Person",
                stair_type: str = "dog_leg") -> Dict[str, Fit]:
    """How each of 1-4BHK fits the bare plot: 'ok', 'tight' (small rooms) or 'no' (a bedroom is dropped)."""
    ctx = _Ctx(plot_w, plot_h, facing, "2BHK", family, building_type, num_floors, "Standard", vastu,
               has_lift, lift_capacity, stair_type)
    return _bhk_options(ctx)


def _bhk_options(ctx: "_Ctx") -> Dict[str, Fit]:
    from floorplan_generator.room_specs import recommend_bhk_for_plot
    order = ["1BHK", "2BHK", "3BHK", "4BHK"]
    cap = order.index(recommend_bhk_for_plot(ctx.W * ctx.H))
    out: Dict[str, Fit] = {}
    for i, b in enumerate(order):
        core = ctx.program(b, quality="Standard", family=_core_family(ctx.family))
        site, _ = _simulate(ctx, core, {}, place=False)
        house, comfy = _house(site), core.target_area * 1.12
        if i > cap or house < core.min_area:
            need = _cents_needed(ctx, lambda c: _bhk_ok(c, b))
            out[b] = Fit(b, "no", "too big for this plot — a bedroom would have to be dropped",
                         unlock=f"Needs about {need:g} cents." if need else "", house_m2=house)
        elif house < COMFORT * comfy:
            out[b] = Fit(b, "tight", "fits, but with small rooms", house_m2=house)
        else:
            out[b] = Fit(b, "ok", "fits well", house_m2=house)
    return out


def apply_fit(components: Dict, fit: SiteFit) -> Dict:
    """The components dict the engine should get: unfit amenities off, values clamped, hints added."""
    out = {}
    for key, cfg in (components or {}).items():
        cfg = dict(cfg or {})
        f = fit.amenities.get(key)
        if f and cfg.get("enabled"):
            if f.status == "no":
                cfg = {"enabled": False, **({"slots": 0} if "slots" in cfg else {}),
                       **({"pct": 0} if "pct" in cfg else {}), **({"size": ""} if "size" in cfg else {})}
            else:
                if f.max_value is not None and "slots" in cfg:
                    cfg["slots"] = min(int(cfg.get("slots") or 1), f.max_value)
                if f.max_value is not None and "pct" in cfg:
                    cfg["pct"] = min(int(cfg.get("pct") or 5), f.max_value)
                if f.choices and cfg.get("size") not in f.choices:
                    cfg["size"] = f.choices[0]
                cfg.update(f.hint)
        out[key] = cfg
    return out


# ── helpers ─────────────────────────────────────────────────────────────

class _CoreFamily:
    """The family without optional rooms: a BHK's fit should not hinge on a study or store."""

    def __init__(self, fam):
        self.adults = getattr(fam, "adults", 2) if fam else 2
        self.children = getattr(fam, "children", 0) if fam else 0
        self.elderly = getattr(fam, "elderly", 0) if fam else 0
        self.guests_frequent = getattr(fam, "guests_frequent", False) if fam else False
        self.needs_pooja_room = getattr(fam, "needs_pooja_room", False) if fam else False
        self.needs_home_office = self.needs_servant_quarter = self.needs_store_room = False


def _core_family(fam):
    return _CoreFamily(fam)


def _bhk_ok(ctx: _Ctx, bhk: str) -> bool:
    from floorplan_generator.room_specs import recommend_bhk_for_plot
    order = ["1BHK", "2BHK", "3BHK", "4BHK"]
    if order.index(bhk) > order.index(recommend_bhk_for_plot(ctx.W * ctx.H)):
        return False
    core = ctx.program(bhk, quality="Standard", family=_core_family(ctx.family))
    site, _ = _simulate(ctx, core, {}, place=False)
    return _house(site) >= max(core.min_area, COMFORT * core.target_area * 1.12)   # rooms of a usable size


def _cents_needed(ctx: _Ctx, ok) -> Optional[float]:
    cents = ctx.W * ctx.H / 40.4686
    for step in range(1, 41):
        c = cents + 0.25 * step
        if ok(ctx.scaled(c / cents)):
            return math.ceil(c * 4) / 4
    return None


def _lifestyle_dropped(ctx: _Ctx, house: float) -> List[str]:
    """Optional rooms (study, store, servant, pooja) the generator's trim step would leave out."""
    from floorplan_generator.space_program import trim_program
    prog = ctx.program(ctx.bhk)
    if house >= prog.min_area:
        return []
    before = {g.primary.name for g in prog.groups}
    trim_program(prog, house)
    return [n for n in sorted(before - {g.primary.name for g in prog.groups})]


LIFT_WHERE = {"inside": "Inside the home", "entrance": "Outside, beside the main door",
              "side": "Outside, at the side of the house"}


def lift_options(plot_w: float, plot_h: float, facing: str = "N", bhk: str = "2BHK", family=None,
                 building_type: str = "Independent House", num_floors: int = 2, vastu: bool = True,
                 lift_capacity: str = "4 Person", stair_type: str = "dog_leg") -> Dict[str, Fit]:
    """
    Where a house's lift can stand (the engine's own reservation, judged with the comfort rule):
      inside   — always: next to the stair, opens into the home on every floor;
      entrance — a tower in the front yard beside the main door (deeper front yard);
      side     — a tower in a side yard with a landing and a side door (a wider side yard).
    An outside tower that would leave the house below COMFORT × its comfortable size is 'no'.
    """
    ctx = _Ctx(plot_w, plot_h, facing, bhk, family, building_type, num_floors, "Standard", vastu,
               True, lift_capacity, stair_type)
    floorwise = "independent" in (building_type or "").lower() and int(num_floors or 1) > 1
    out = {"inside": Fit("inside", "ok", "opens onto the stair lobby on every floor, next to the stair" if floorwise
                         else "opens into the home on every floor, next to the stair")}
    inside = ctx.program(bhk)
    if not inside.lift:
        return out
    base = _house(_simulate(ctx, inside, {}, place=False)[0])
    from floorplan_generator.vastu_engine import plan_sides_for_facing
    sides = plan_sides_for_facing(ctx.facing)
    words = {"N": "North", "E": "East", "S": "South", "W": "West"}
    for pos in ("entrance", "side"):
        c2 = _Ctx.__new__(_Ctx)
        c2.__dict__.update(ctx.__dict__)
        c2.lift_position = pos
        prog = c2.program(bhk)
        site = _simulate(c2, prog, {}, place=False)[0]
        if site.lift_side is None or (pos == "side" and site.lift_side == "top"):
            why = ("the front yard cannot take the lift tower and its landing" if pos == "entrance"
                   else "no side yard is wide enough for the lift tower (about "
                        f"{SitePlanner.lift_side_need(_lift_out(prog)):.1f} m)")
            out[pos] = Fit(pos, "no", why)
            continue
        m2 = _house(site)
        comfy = prog.target_area * 1.12
        shaft = prog.lift.shaft_w * prog.lift.shaft_d          # the inside lift's floor space the rooms get back
        loss = max(0.0, base - shaft - m2)
        if m2 < prog.min_area or (loss > LOSS * base and m2 < COMFORT * comfy):
            out[pos] = Fit(pos, "no", f"the tower's yard would take {_sqft(loss)} from the house — every room "
                                      "would be smaller than comfortable", house_m2=m2, loss_m2=loss)
            continue
        where = ("in the front yard, beside the main door" if pos == "entrance"
                 else f"in the {words.get(sides.get(site.lift_side, ''), '')} side yard")
        if loss > LOSS * base and m2 < TIGHT * comfy:
            out[pos] = Fit(pos, "tight", f"fits {where}, but the house gets {_sqft(loss)} smaller",
                           house_m2=m2, loss_m2=loss)
        else:
            out[pos] = Fit(pos, "ok", f"fits {where} — each floor reached without passing through another",
                           house_m2=m2, loss_m2=loss)
    return out

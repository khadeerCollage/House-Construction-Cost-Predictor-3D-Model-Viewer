"""
Land value & property worth — where the plot is decides what it is worth
=======================================================================
What owners (and their parents) ask first is "how much will this house be worth?". The
answer is mostly the land, and land is priced street by street: in Chilakaluripet (Palnadu,
Andhra Pradesh) an inner residential street is ₹6–8 lakh a cent, the main road ₹10–11 lakh,
the town outskirts ₹3–4 lakh. This module estimates it from the plot's location:

  1. Official guideline value (the government's minimum value for registration) of the town,
     where we carry it: every Andhra Pradesh and Telangana town from IGRS market values
     FY 2025-26 (data/land_rates_ap_ts.json).
  2. × a street-position factor calibrated on a real town: Chilakaluripet(U) guideline
     ₹17,000/sq yd vs local market rates — inner street ≈ 0.85×, main road ≈ 1.3×,
     outskirts layout ≈ 0.43×, village ≈ 0.25× (with low–high bands).
  3. Elsewhere: the state's typical district-town inner-street rate × town size × street
     position (indicative — the UI says so and asks for the local rate).
  4. The owner's own local rate (per cent / sq yd / sq ft / ankanam / guntha / ground /
     marla), when given, always wins.

It also gives the state's registration charges (stamp duty + transfer duty / cess +
registration fee, on the higher of market and guideline value) and the construction
location index (state × town size) that CostEngine uses for the building.

All rates are 2025-26 planning figures, not valuations — the UI shows the basis and
confidence of every number and where to check the official value.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os

SQFT_PER_M2 = 10.7639
UNITS = {   # unit → square feet
    "per cent": 435.6, "per sq yd": 9.0, "per sq ft": 1.0, "per ankanam": 72.0,
    "per guntha": 1089.0, "per ground": 2400.0, "per marla": 272.25, "per sq m": SQFT_PER_M2,
}

# Street position. guideline = factor on the town's official guideline value (calibrated on
# Chilakaluripet: market ₹6–8 L/cent inner, ₹10–11 L main road, ₹3–4 L outskirts vs ₹17,000/sq yd);
# estimate = factor on the state's typical inner-street rate (same ratios: main 1.5×, outskirts 0.5×).
AREA_TYPES = [
    # key,        label,                                 icon, guideline (lo, mid, hi), estimate (lo, mid, hi)
    ("main_road", "Main road / bazaar frontage",          "🛣️", (1.20, 1.30, 1.40), (1.35, 1.50, 1.65)),
    ("inner",     "Inner street / residential colony",    "🏘️", (0.73, 0.85, 0.97), (0.85, 1.00, 1.15)),
    ("outskirts", "Town outskirts / new layout",          "🌾", (0.36, 0.43, 0.49), (0.42, 0.50, 0.58)),
    ("village",   "Village (gram panchayat area)",        "🏡", (0.18, 0.25, 0.32), (0.18, 0.25, 0.32)),
]
AREA = {a[0]: a for a in AREA_TYPES}

TOWN_CLASSES = {   # unknown towns: share of the state's district-HQ rate; construction index factor
    "city": ("Big city / district headquarters", 1.00, 1.00),
    "town": ("Town (municipality / mandal or taluk HQ)", 0.65, 0.95),
    "village": ("Village", 0.20, 0.88),
}

# State data (2025-26): stamp duty % (men, women), registration %, other duty % (transfer duty / cess),
# registration cap ₹, construction index vs India Tier-2 average, typical inner-street land rate in a
# district-HQ town (₹/sq ft, indicative), the unit people quote land in, and the official value portal.
# Stamp/registration: compiled from 2025-26 state tables (Godrej Capital, housing.com, Bajaj Finserv,
# IGRS AP/TS); several states use slabs or urban/rural splits — verify at the sub-registrar.
STATES: Dict[str, Dict] = {
    "Andhra Pradesh":    dict(stamp=(5.0, 5.0), reg=1.0, other=1.5, other_label="transfer duty", cap=None,
                              build=0.97, land_sqft=1900, unit="per sq yd", alt_unit="per cent",
                              portal=("IGRS Andhra Pradesh — Market Value", "https://registration.ap.gov.in")),
    "Telangana":         dict(stamp=(4.0, 4.0), reg=0.5, other=1.5, other_label="transfer duty", cap=None,
                              build=0.98, land_sqft=1900, unit="per sq yd", alt_unit="per sq ft",
                              rural=dict(stamp=(5.5, 5.5), reg=2.0, other=0.0),
                              portal=("IGRS Telangana — Market Value", "https://registration.telangana.gov.in")),
    "Tamil Nadu":        dict(stamp=(7.0, 7.0), reg=2.0, other=0.0, cap=None, build=1.00, land_sqft=2200,
                              unit="per sq ft", alt_unit="per cent",
                              portal=("TNREGINET — Guideline Value", "https://tnreginet.gov.in")),
    "Kerala":            dict(stamp=(8.0, 8.0), reg=2.0, other=0.0, cap=None, build=1.12, land_sqft=2800,
                              unit="per cent", alt_unit="per sq ft",
                              portal=("Kerala Registration — Fair Value", "https://keralaregistration.gov.in")),
    "Karnataka":         dict(stamp=(5.0, 5.0), reg=1.0, other=0.5, other_label="cess & surcharge", cap=None,
                              build=1.02, land_sqft=2000, unit="per sq ft", alt_unit="per sq yd",
                              portal=("Kaveri Online — Guidance Value", "https://kaveri.karnataka.gov.in")),
    "Maharashtra":       dict(stamp=(5.0, 4.0), reg=1.0, other=1.0, other_label="metro / local body cess",
                              cap=30000, build=1.05, land_sqft=2200, unit="per sq ft", alt_unit="per guntha",
                              portal=("IGR Maharashtra — Ready Reckoner", "https://igrmaharashtra.gov.in")),
    "Goa":               dict(stamp=(4.5, 4.5), reg=3.0, other=0.0, cap=None, build=1.15, land_sqft=3500,
                              unit="per sq m", alt_unit="per sq ft",
                              portal=("Goa Registration", "https://egov.goa.gov.in")),
    "Gujarat":           dict(stamp=(4.9, 4.9), reg=1.0, other=0.0, cap=None, build=0.97, land_sqft=2200,
                              unit="per sq yd", alt_unit="per sq ft",
                              portal=("Garvi Gujarat — Jantri", "https://garvi.gujarat.gov.in")),
    "Rajasthan":         dict(stamp=(6.0, 5.0), reg=1.0, other=0.0, cap=None, build=0.92, land_sqft=1600,
                              unit="per sq yd", alt_unit="per sq ft",
                              portal=("IGRS Rajasthan — DLC rate", "https://epanjiyan.rajasthan.gov.in")),
    "Madhya Pradesh":    dict(stamp=(7.5, 6.5), reg=3.0, other=0.0, cap=None, build=0.90, land_sqft=1400,
                              unit="per sq ft", alt_unit="per sq m",
                              portal=("MP Sampada — Guideline", "https://sampada.mpigr.gov.in")),
    "Chhattisgarh":      dict(stamp=(5.0, 4.0), reg=4.0, other=0.0, cap=None, build=0.88, land_sqft=1200,
                              unit="per sq ft", alt_unit="per sq m",
                              portal=("CG Registration", "https://igrs.cg.gov.in")),
    "Uttar Pradesh":     dict(stamp=(7.0, 6.0), reg=1.0, other=0.0, cap=None, build=0.90, land_sqft=1800,
                              unit="per sq yd", alt_unit="per sq ft",
                              portal=("IGRS Uttar Pradesh — Circle rate", "https://igrsup.gov.in")),
    "Bihar":             dict(stamp=(6.0, 5.7), reg=2.0, other=0.0, cap=None, build=0.90, land_sqft=2000,
                              unit="per sq ft", alt_unit="per sq yd",
                              portal=("Bihar Registration — MVR", "https://bhumijankari.bihar.gov.in")),
    "Jharkhand":         dict(stamp=(4.0, 4.0), reg=3.0, other=0.0, cap=None, build=0.90, land_sqft=1500,
                              unit="per sq ft", alt_unit="per sq yd",
                              portal=("Jharkhand Registration", "https://jharnibandhan.jharkhand.gov.in")),
    "West Bengal":       dict(stamp=(6.0, 6.0), reg=1.0, other=0.0, cap=None, build=0.95, land_sqft=2000,
                              unit="per sq ft", alt_unit="per sq yd",
                              portal=("WB Registration — Market Value", "https://wbregistration.gov.in")),
    "Odisha":            dict(stamp=(5.0, 4.0), reg=2.0, other=0.0, cap=None, build=0.92, land_sqft=1600,
                              unit="per sq ft", alt_unit="per sq yd",
                              portal=("IGR Odisha — Benchmark value", "https://igrodisha.gov.in")),
    "Assam":             dict(stamp=(8.25, 6.0), reg=1.0, other=0.0, cap=None, build=1.00, land_sqft=1800,
                              unit="per sq ft", alt_unit="per sq yd",
                              portal=("Assam Registration", "https://igr.assam.gov.in")),
    "Punjab":            dict(stamp=(7.0, 5.0), reg=1.0, other=0.0, cap=None, build=0.98, land_sqft=2500,
                              unit="per sq yd", alt_unit="per marla",
                              portal=("Punjab Revenue — Collector rate", "https://revenue.punjab.gov.in")),
    "Haryana":           dict(stamp=(7.0, 5.0), reg=1.0, other=0.0, cap=50000, build=1.02, land_sqft=2800,
                              unit="per sq yd", alt_unit="per marla",
                              portal=("Jamabandi Haryana — Collector rate", "https://jamabandi.nic.in")),
    "Himachal Pradesh":  dict(stamp=(6.0, 4.0), reg=2.0, other=0.0, cap=None, build=1.12, land_sqft=2500,
                              unit="per sq ft", alt_unit="per sq m",
                              portal=("HP Registration", "https://himachal.nic.in")),
    "Uttarakhand":       dict(stamp=(5.0, 3.75), reg=2.0, other=0.0, cap=None, build=1.08, land_sqft=2200,
                              unit="per sq ft", alt_unit="per sq m",
                              portal=("Uttarakhand Registration", "https://registration.uk.gov.in")),
    "Jammu & Kashmir":   dict(stamp=(5.0, 3.0), reg=1.2, other=0.0, cap=None, build=1.10, land_sqft=2000,
                              unit="per marla", alt_unit="per sq ft",
                              portal=("J&K Registration", "https://jkrevenue.nic.in")),
    "Delhi":             dict(stamp=(6.0, 4.0), reg=1.0, other=0.0, cap=None, build=1.25, land_sqft=25000,
                              unit="per sq yd", alt_unit="per sq ft",
                              portal=("Delhi Revenue — Circle rate", "https://revenue.delhi.gov.in")),
    "Chandigarh":        dict(stamp=(6.0, 4.0), reg=1.0, other=0.0, cap=None, build=1.08, land_sqft=12000,
                              unit="per sq yd", alt_unit="per sq ft",
                              portal=("Chandigarh Estate Office", "https://chandigarh.gov.in")),
    "Puducherry":        dict(stamp=(7.0, 7.0), reg=4.0, other=0.0, cap=None, build=1.00, land_sqft=3000,
                              unit="per sq ft", alt_unit="per cent",
                              portal=("Puducherry Registration", "https://registration.py.gov.in")),
    "Sikkim":            dict(stamp=(5.0, 5.0), reg=1.0, other=0.0, cap=None, build=1.15, land_sqft=2000,
                              unit="per sq ft", alt_unit="per sq m", portal=("Sikkim Land Revenue", "https://sikkim.gov.in")),
    "Tripura":           dict(stamp=(5.0, 5.0), reg=1.0, other=0.0, cap=None, build=1.08, land_sqft=1500,
                              unit="per sq ft", alt_unit="per sq m", portal=("Tripura Registration", "https://tripura.gov.in")),
    "Meghalaya":         dict(stamp=(6.0, 6.0), reg=1.0, other=0.0, cap=None, build=1.10, land_sqft=1800,
                              unit="per sq ft", alt_unit="per sq m", portal=("Meghalaya Registration", "https://meghalaya.gov.in")),
    "Manipur":           dict(stamp=(7.0, 7.0), reg=3.0, other=0.0, cap=None, build=1.10, land_sqft=1500,
                              unit="per sq ft", alt_unit="per sq m", portal=("Manipur Registration", "https://manipur.gov.in")),
    "Mizoram":           dict(stamp=(9.0, 9.0), reg=1.0, other=0.0, cap=None, build=1.12, land_sqft=1500,
                              unit="per sq ft", alt_unit="per sq m", portal=("Mizoram Registration", "https://mizoram.gov.in")),
    "Nagaland":          dict(stamp=(8.25, 8.25), reg=1.25, other=0.0, cap=None, build=1.12, land_sqft=1500,
                              unit="per sq ft", alt_unit="per sq m", portal=("Nagaland Registration", "https://nagaland.gov.in")),
    "Arunachal Pradesh": dict(stamp=(6.0, 6.0), reg=1.0, other=0.0, cap=None, build=1.15, land_sqft=1200,
                              unit="per sq ft", alt_unit="per sq m", portal=("Arunachal Registration", "https://arunachalpradesh.gov.in")),
}
DEFAULT_STATE = "Andhra Pradesh"

# Metro cities: inner residential plot rate in the city (₹/sq ft, indicative 2025-26) and the
# construction index CostEngine already uses for the city.
CITIES = {
    "Delhi": ("Delhi", 25000, 1.30), "Mumbai": ("Maharashtra", 30000, 1.55), "Pune": ("Maharashtra", 6500, 1.18),
    "Bengaluru": ("Karnataka", 8000, 1.18), "Chennai": ("Tamil Nadu", 8000, 1.10),
    "Kolkata": ("West Bengal", 5000, 1.05), "Ahmedabad": ("Gujarat", 5000, 1.02),
    "Kochi / Ernakulam": ("Kerala", 4500, 1.25), "Thiruvananthapuram": ("Kerala", 4000, 1.18),
    "Coimbatore": ("Tamil Nadu", 3500, 1.00), "Jaipur": ("Rajasthan", 3500, 1.03), "Lucknow": ("Uttar Pradesh", 3500, 1.02),
    "Indore": ("Madhya Pradesh", 3500, 0.97), "Chandigarh": ("Chandigarh", 12000, 1.10),
}

_DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "land_rates_ap_ts.json")
_TOWNS_CACHE: Optional[Dict] = None


def official_towns() -> Dict:
    """{state: {district: {hq, district_range, district_median, towns: [{town, rate, range, mandal}]}}}."""
    global _TOWNS_CACHE
    if _TOWNS_CACHE is None:
        try:
            with open(_DATA_FILE, encoding="utf-8") as f:
                raw = json.load(f)
            _TOWNS_CACHE = {"Andhra Pradesh": raw.get("andhra_pradesh", {}), "Telangana": raw.get("telangana", {})}
            year = raw.get("year", "FY 2025-26")
            _TOWNS_CACHE["_meta"] = {"year": year if isinstance(year, dict) else {"Andhra Pradesh": year},
                                     "source": raw.get("source", ""),
                                     "agency": set(raw.get("agency_districts", []))}
        except (OSError, ValueError):
            _TOWNS_CACHE = {"_meta": {"year": {}, "source": "", "agency": set()}}
    return _TOWNS_CACHE


def districts(state: str) -> List[str]:
    return sorted(official_towns().get(state, {}).keys())


def towns(state: str, district: str) -> List[str]:
    d = official_towns().get(state, {}).get(district) or {}
    names = [t["town"] for t in d.get("towns", [])]
    hq = d.get("hq")
    if hq in names:                     # headquarters first
        names.remove(hq)
        names.insert(0, hq)
    return names


def rate_text(amount: float) -> str:
    """₹14,450 · ₹6.99 lakh · ₹1.20 crore — the way Indian buyers say land rates."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} crore"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} lakh"
    s = f"{int(round(amount))}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + parts + [tail])
    return f"₹{s}"


def per_unit(rate_sqft: float, unit: str) -> float:
    return rate_sqft * UNITS[unit]


def to_sqft(rate: float, unit: str) -> float:
    return rate / UNITS[unit]


@dataclass
class LandEstimate:
    state: str
    place: str                       # "Chilakaluripet, Palnadu"
    area_type: str
    rate_sqft: Tuple[float, float, float]      # low, mid, high (₹ per sq ft)
    value: Tuple[float, float, float]          # for the plot
    plot_sqft: float
    basis: str                       # how the rate was found
    confidence: str                  # "your rate" | "official value" | "indicative"
    guideline_sqft: Optional[float] = None     # official value, when known
    construction_index: float = 1.0
    unit: str = "per sq yd"
    alt_unit: str = "per cent"
    portal: Tuple[str, str] = ("", "")
    notes: List[str] = field(default_factory=list)


def estimate_land(plot_area_m2: float, state: str = DEFAULT_STATE, district: str = "", town: str = "",
                  area_type: str = "inner", town_class: str = "town",
                  known_rate: Optional[float] = None, known_unit: str = "per cent") -> LandEstimate:
    """Land value of the plot from its location (see module docstring for the method)."""
    st = STATES.get(state) or STATES[DEFAULT_STATE]
    plot_sqft = plot_area_m2 * SQFT_PER_M2
    key, label, _, g_f, e_f = AREA.get(area_type, AREA["inner"])
    notes: List[str] = []
    guideline = None
    place = ", ".join(p for p in (town, district) if p) or state
    t_class = town_class

    # official guideline value of the town (AP / Telangana data)
    d = official_towns().get(state, {}).get(district) or {}
    rec = next((t for t in d.get("towns", []) if t["town"] == town), None)
    if rec and rec.get("rate"):
        guideline = rec["rate"] / 9.0                      # ₹/sq yd → ₹/sq ft
        lo, mid, hi = (guideline * f for f in g_f)
        meta = official_towns()["_meta"]
        year = meta["year"].get(state, "FY 2025-26")
        basis = (f"Official guideline value of {town} ₹{rec['rate']:,.0f}/sq yd ({year}) × {g_f[1]:g} for "
                 f"{label.lower()} (street factor calibrated on Chilakaluripet market rates)")
        confidence = "official value"
        # district headquarters and towns valued like cities get city building rates
        t_class = "city" if (town == d.get("hq") or rec["rate"] >= 20000) else "town"
        if rec.get("range"):
            notes.append(f"Official street-wise values in {town} run ₹{rec['range'][0]:,.0f}–"
                         f"₹{rec['range'][1]:,.0f} per sq yd; the exact one depends on your door / survey number.")
        if rec["rate"] >= 30000:
            notes.append("In big cities sale prices often run well above the official value — for a city plot, "
                         "enter the rate you are being quoted for the most accurate figure.")
        if district in meta["agency"]:
            notes.append(f"{district} is a Scheduled (agency) area: land transfer is restricted and official values "
                         "are very low — check eligibility before buying.")
    elif town in CITIES:
        _, city_rate, _ = CITIES[town]
        lo, mid, hi = (city_rate * f for f in e_f)
        basis = f"Typical {town} residential plot rate ₹{city_rate:,.0f}/sq ft (indicative) × {e_f[1]:g} for {label.lower()}"
        confidence = "indicative"
        t_class = "metro"
    else:
        base = st["land_sqft"]
        if d.get("towns"):                                 # unlisted town in a known district
            hq = next((t for t in d["towns"] if t["town"] == d.get("hq")), d["towns"][0])
            base = hq["rate"] / 9.0 * 0.85
            where = f"{district} district (HQ {hq['town']} official ₹{hq['rate']:,.0f}/sq yd)"
        else:
            where = f"{state} district towns (₹{base:,.0f}/sq ft, indicative)"
        share = TOWN_CLASSES.get(town_class, TOWN_CLASSES["town"])[1]
        lo, mid, hi = (base * share * f for f in e_f)
        basis = (f"Estimated from {where} × {share:g} for: {TOWN_CLASSES.get(town_class, TOWN_CLASSES['town'])[0]}"
                 f" × {e_f[1]:g} for {label.lower()}")
        confidence = "indicative"

    if known_rate:
        r = to_sqft(float(known_rate), known_unit)
        lo, mid, hi = r * 0.95, r, r * 1.05
        basis = f"Your local rate ₹{float(known_rate):,.0f} {known_unit}"
        confidence = "your rate"
        if guideline and r < guideline * 0.6:
            notes.append("Your rate is well below the official guideline value — stamp duty is still charged on the "
                         "guideline value.")

    if t_class == "metro":
        build = CITIES[town][2]
    else:
        build = st["build"] * {"city": 1.00, "town": 0.95, "village": 0.88}.get(
            "village" if key == "village" else t_class, 0.95)
    return LandEstimate(
        state=state, place=place, area_type=key, rate_sqft=(lo, mid, hi),
        value=(lo * plot_sqft, mid * plot_sqft, hi * plot_sqft), plot_sqft=plot_sqft, basis=basis,
        confidence=confidence, guideline_sqft=guideline, construction_index=round(build, 3),
        unit=st["unit"], alt_unit=st["alt_unit"], portal=st["portal"], notes=notes)


def registration_charges(land: LandEstimate, buyer_female: bool = False) -> Dict:
    """Stamp duty + transfer duty / cess + registration fee on the higher of market and guideline value."""
    st = STATES.get(land.state) or STATES[DEFAULT_STATE]
    rules = st
    if land.area_type == "village" and st.get("rural"):
        rules = dict(st, **st["rural"])
    market = land.value[1]
    # the town's representative official value is taken as its inner-street value; official street
    # values rise and fall with the street like market rates do (main road higher, outskirts lower)
    street = AREA[land.area_type][3][1] / AREA["inner"][3][1]
    guide = (land.guideline_sqft or 0) * street * land.plot_sqft
    base = max(market, guide)
    stamp_pct = rules["stamp"][1 if buyer_female else 0]
    reg_pct, other_pct = rules["reg"], rules.get("other", 0.0)
    reg = base * reg_pct / 100
    if st.get("cap"):
        reg = min(reg, st["cap"])
    items = [("Stamp duty", stamp_pct, base * stamp_pct / 100)]
    if other_pct:
        items.append((st.get("other_label", "other duty").capitalize(), other_pct, base * other_pct / 100))
    items.append(("Registration fee", reg_pct, reg))
    return {"base": base, "on": "guideline value" if guide > market else "market value",
            "items": [{"label": l, "pct": p, "amount": round(a)} for l, p, a in items],
            "total": round(sum(a for _, _, a in items)), "pct": round(sum(p for _, p, _ in items), 2),
            "female_saves": round(base * (rules["stamp"][0] - rules["stamp"][1]) / 100)}


def property_worth(land: LandEstimate, est: Dict, already_own: bool = False,
                   buyer_female: bool = False) -> Dict:
    """
    The money report: what the family spends (land + registration + house) and what the finished
    property is worth (land at market value + the new building at its construction cost — the
    valuers' land-and-building method; soft costs and duties are money spent, not value added).
    """
    reg = registration_charges(land, buyer_female)
    house_total = est.get("total_cost", 0.0)
    building_value = est.get("construction_cost", house_total)
    spend_land = 0.0 if already_own else land.value[1]
    spend_reg = 0.0 if already_own else reg["total"]
    worth = tuple(v + building_value for v in land.value)
    return {
        "land": land.value, "registration": reg, "house_total": house_total,
        "building_value": building_value,
        "investment": spend_land + spend_reg + house_total,
        "worth": worth,
        "already_own": already_own,
        "per_sqft_builtup": house_total / max(1.0, est.get("area_sqft", 1.0)),
    }

"""
Wish Parser — plain-language house wishes → design intents (offline, rule-based)
================================================================================
Owners describe changes the way they speak, not the way architects write:

    "please make hall is big"            "kichen to north east"
    "master room attached bathroom other side"
    "hall too small"  (a complaint → make it bigger)
    "no kitchen in north east"           "rasoi uttar mein karo"
    "adukkala vadakku kizhakku"          "swap kids room and study, pooja near hall"

Pipeline (no external service, no ML model):
  1. split into clauses (. ; , ! ? newline, "then", "also", and "and" between two wishes)
  2. tokenise (works for Latin and Indic scripts)
  3. spelling repair: unknown words are matched to the vocabulary with difflib
     (similarity ≥ 0.75–0.8); every repair is reported back ("kichen → kitchen")
  4. phrase lexicon (longest match first): rooms, actions, directions, plan-sides,
     intensifiers, negation — English, Indian English, Hindi / Malayalam / Tamil /
     Telugu transliterations and a few native-script words
  5. frame → intent: action + room(s) + target, with complaint inversion
     ("too small" ⇒ bigger), negation ("not in NE" ⇒ anywhere else reachable),
     corners from two sides ("north … east" ⇒ NE), bedroom numbers and
     attached-bath owners ("master bathroom")
  6. confidence: high (clear) · medium (spelling repaired / inferred) → added
     straight away; low (ambiguous) → "did you mean …" suggestions only

Output per clause: {clause, intents, suggestions, confidence, corrections, note}.
"""

import difflib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- lexicon
_P = {
    # ---- rooms
    "R:living": ["living", "living room", "livingroom", "hall", "haal", "hal", "drawing room", "drawingroom", "lounge",
                 "sitting room", "sitting", "front room", "visitors room", "baithak", "baithak khana",
                 "swikarana muri", "sweekarana muri", "koodam", "मुख्य कमरा", "हॉल", "ഹാൾ"],
    "R:kitchen": ["kitchen", "kitchenette", "cooking", "cook room", "cooking room", "cooking area", "rasoi",
                  "rasoighar", "rasoda", "adukkala", "adukala", "samayal arai", "samayalarai", "samayal",
                  "vantagadi", "vantillu", "रसोई", "रसोईघर", "അടുക്കള", "சமையலறை"],
    "R:master": ["master", "master bedroom", "masterbedroom", "master room", "master bed", "main bedroom",
                 "main room", "big bedroom", "parents room", "parent room", "parents bedroom", "my room",
                 "our room", "couple room", "my bedroom", "our bedroom"],
    "R:child": ["children", "childrens", "children room", "childrens room", "childrens bedroom", "child bedroom", "kids", "kid", "kids room",
                "kidsroom", "kid room", "child room", "child", "son room", "daughter room", "baby room",
                "children bedroom", "kids bedroom"],
    "R:guest": ["guest", "guest room", "guest bedroom", "guestroom", "visitor room"],
    "R:elder": ["elder", "elders", "elderly", "elders bedroom", "grandparents", "grand parents", "grandpa", "grandma",
                "old parents", "elders room", "senior room"],
    "R:bedroom": ["bedroom", "bed room", "bed", "kidappumuri", "kidapumuri", "kidakka muri", "shayan kaksh",
                  "shayankaksh", "padukkai arai", "padukkaiarai", "padaka gadi", "shayan kaksha", "शयनकक्ष",
                  "बेडरूम", "കിടപ്പുമുറി"],
    "R:attached_bath": ["attached bathroom", "attached bath", "attached toilet", "attached washroom",
                        "attach bathroom", "attach toilet", "ensuite", "en suite", "attached"],
    "R:attached_bath@master": ["master bathroom", "master bath", "master toilet", "master washroom"],
    "R:attached_bath@child": ["kids bathroom", "kids toilet", "children bathroom", "children toilet"],
    "R:bath": ["bathroom", "bath room", "bath", "toilet", "toilets", "washroom", "wash room", "restroom", "wc",
               "latrine", "lavatory", "loo", "common bathroom", "common toilet", "common bath", "kulimuri",
               "kuli muri", "shauchalay", "shauchalaya", "snanghar", "kuliyal arai", "bathroomu", "बाथरूम",
               "शौचालय", "കുളിമുറി", "குளியலறை"],
    "R:pooja": ["pooja", "puja", "pooja room", "puja room", "poojaroom", "pujaroom", "prayer", "prayer room",
                "mandir", "temple", "god room", "deity room", "shrine", "pooja ghar", "devghar", "pooja muri",
                "पूजा", "पूजाघर", "പൂജാമുറി", "പൂജ"],
    "R:study": ["study", "study room", "office", "home office", "work room", "workroom", "library",
                "computer room", "reading room", "office room"],
    "R:store": ["store", "store room", "storeroom", "storage", "godown", "box room", "store house"],
    "R:utility": ["utility", "utility room", "wash area", "washing area", "laundry", "work area", "washing",
                  "wash", "utility area"],
    "R:dining": ["dining", "dining room", "diningroom", "dining hall", "dining area", "eating area",
                 "dinning", "dinning room"],
    "R:stairs": ["stairs", "staircase", "stair", "stair case", "steps", "sopanam", "seedi", "सीढ़ी"],
    "R:lift": ["lift", "elevator"],
    "R:servant": ["servant", "servant room", "maid", "maid room", "driver room", "helper room", "staff room"],
    # ---- actions
    "A:bigger": ["bigger", "big", "large", "larger", "huge", "spacious", "wide", "wider", "enlarge", "expand",
                 "increase", "extend", "more space", "more area", "more room", "grow", "extra space",
                 "make big", "more big", "bada", "badi", "bara", "bade", "valiya", "valuthu", "valuthakku",
                 "periya", "perisu", "pedda", "बड़ा", "बड़ी", "वलिया", "വലിയ", "വലുത്"],
    "A:smaller": ["smaller", "small", "tiny", "reduce", "decrease", "shrink", "compact", "less space",
                  "less area", "minimize", "minimise", "cut down", "chota", "chhota", "choti", "chhoti",
                  "cheriya", "cheruthu", "chinna", "siriya", "chinnadi", "छोटा", "छोटी", "ചെറിയ", "ചെറുത്"],
    "A:swap": ["swap", "exchange", "interchange", "switch", "replace", "change places", "change place",
               "trade places", "inter change", "adal badal", "maatti vekkuka"],
    "A:adjacent": ["next to", "near", "nearby", "beside", "besides", "adjacent", "adjacent to", "close to",
                   "closer to", "attached to", "connect", "connected", "connected to", "join", "joined",
                   "along with", "near to", "next", "paas", "pass", "aduthu", "arukil", "pakkathil", "pakkam"],
    "A:lock": ["keep", "lock", "fix", "fixed", "same place", "dont move", "do not move", "dont change",
               "do not change", "dont touch", "do not touch", "leave", "as it is", "as is", "no change",
               "stay", "same", "untouched", "mat hilao", "mat badlo", "maattanda"],
    "A:opposite": ["opposite", "other side", "another side", "other end", "opposite side", "reverse side",
                   "flip", "ulta", "ulta side", "doosri taraf", "maru vasham", "marupuram", "ethir"],
    "A:mirror": ["mirror", "mirror image", "mirrored", "flip plan", "flip the plan", "flip house",
                 "reverse plan", "reverse the plan", "flip layout"],
    "A:move": ["move", "shift", "put", "place", "relocate", "transfer", "change", "set", "keep in",
               "badlo", "badal", "maattuka", "maattu", "maattanam", "maathu", "maatru", "mattu", "hatao",
               "rakho", "rakh", "vekkuka", "vaikkuka"],
    # ---- directions (compass corners / sides)
    "D:NE": ["north east", "northeast", "north-east", "ne", "n e", "ishan", "ishanya", "eesanya", "esanya",
             "eshanya", "isanya", "uttar purv", "eesana moola", "ishana", "ईशान", "ഈശാനം"],
    "D:SE": ["south east", "southeast", "south-east", "s e", "agni", "agneya", "aagneya", "agney",
             "dakshin purv", "agni moola", "आग्नेय", "ആഗ്നേയം"],
    "D:SW": ["south west", "southwest", "south-west", "sw", "s w", "nairutya", "nairuthi", "nairitya", "niruthi",
             "nairuti", "kanni moola", "kannimoola", "kanni", "dakshin paschim", "नैऋत्य", "കന്നിമൂല"],
    "D:NW": ["north west", "northwest", "north-west", "nw", "n w", "vayu", "vayavya", "vayavyam", "vaayu",
             "uttar paschim", "vayu moola", "वायव्य", "വായുകോൺ"],
    "D:CENTER": ["center", "centre", "middle", "central", "brahmasthan", "brahmasthanam", "madhya", "naduvil",
                 "nadu", "बीच", "നടുവിൽ"],
    "S:N": ["north", "northern", "north side", "uttar", "uttara", "vadakku", "vadakk", "vadakke", "uttaram",
            "उत्तर", "വടക്ക്", "வடக்கு"],
    "S:E": ["east", "eastern", "east side", "purab", "poorv", "purv", "poorab", "kizhakku", "kizhakk",
            "kizhakke", "thoorpu", "toorpu", "पूर्व", "കിഴക്ക്", "கிழக்கு"],
    "S:S": ["south", "southern", "south side", "dakshin", "dakshina", "thekku", "thekk", "thekke", "therku",
            "dakshinam", "दक्षिण", "തെക്ക്", "தெற்கு"],
    "S:W": ["west", "western", "west side", "paschim", "pashchim", "padinjaru", "padinjar", "padinjare",
            "merku", "padamara", "पश्चिम", "പടിഞ്ഞാറ്", "மேற்கு"],
    "P:front": ["front", "front side", "road side", "roadside", "entrance side", "gate side", "facing side",
                "street side", "munbil", "munnil", "aage", "samne", "munpil"],
    "P:back": ["back", "rear", "backside", "back side", "behind", "pinnil", "pichhe", "peeche", "pinpuram"],
    "P:left": ["left", "left side", "baayen", "baaye", "idathu", "idath", "idathe"],
    "P:right": ["right", "right side", "daayen", "daaye", "valathu", "valath", "valathe"],
    # ---- modifiers
    "M:too": ["too", "so", "too much", "not enough", "not big enough"],
    "M:neg": ["not", "no", "dont", "don't", "do not", "never", "without", "nahi", "nahin", "mat", "venda",
              "vendaa", "vendam", "illa", "not in", "no need"],
    "M:want": ["want", "wants", "need", "needs", "make", "should", "like", "please", "pls", "plz", "kindly",
               "wish", "chahiye", "chaiye", "venam", "venum", "vendum", "aakkuka", "aakku", "aakkanam",
               "karo", "kardo", "karna"],
    "M:is": ["is", "are", "looks", "look", "feels", "feel", "seems", "seem", "hai", "aanu"],
    "X:plan": ["plan", "house", "home", "layout", "design", "whole house", "full house", "entire house",
               "veedu", "ghar", "makaan", "मकान", "घर", "വീട്"],
    "C:and": ["and", "aur", "um", "plus", "&"],
    # ---- wishes the planner cannot do (answered with a helpful note instead of a wrong guess)
    "U:merge": ["remove", "delete", "demolish", "break", "merge", "combine", "open plan", "open kitchen",
                "remove wall", "remove the wall", "no wall", "without wall", "hatao wall"],
    "U:site": ["garden", "lawn", "parking", "car", "car porch", "porch", "pool", "swimming pool", "sump",
               "tank", "gate", "compound", "bike", "generator", "security"],
}

_NUM = {"1": 1, "2": 2, "3": 3, "4": 4, "one": 1, "two": 2, "three": 3, "four": 4, "first": 1, "second": 2,
        "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

_STOP = {
    "i", "me", "we", "us", "you", "your", "it", "its", "the", "a", "an", "to", "in", "into", "on", "at", "of",
    "for", "from", "by", "with", "or", "but", "can", "could", "would", "will", "shall", "may", "might", "must",
    "do", "does", "did", "done", "get", "give", "have", "has", "had", "be", "am", "was", "were", "been", "this",
    "that", "these", "those", "there", "here", "side", "room", "rooms", "area", "place", "part", "portion",
    "corner", "all", "then", "just", "only", "more", "much", "bit", "some", "any", "thing", "my", "our", "also",
    "me", "mein", "ko", "ka", "ki", "ke", "se", "par", "mujhe", "hume", "humko", "bhi", "wala", "wali", "il",
    "inte", "ude", "nu", "oru", "ah", "pa", "the", "size", "sized", "now", "currently", "please", "okay", "ok",
    "zone", "direction", "towards", "toward", "up", "down", "again", "position", "location",
    "very", "bahut", "valare", "romba", "chala", "jyada", "zyada", "really", "quite",
    # building words that must never be "corrected" into a room name
    "wall", "walls", "window", "windows", "door", "doors", "floor", "roof", "plot", "land", "road", "gate",
    "light", "air", "sun", "cost", "price", "budget", "money", "tree",
    "where", "what", "which", "when", "how", "why", "about", "around", "between", "inside", "outside",
}

_PHRASES: Dict[Tuple[str, ...], str] = {}
for _tag, _items in _P.items():
    for _ph in _items:
        _PHRASES[tuple(_ph.lower().split())] = _tag
_VOCAB = sorted({w for ph in _PHRASES for w in ph if len(w) >= 3 and not w.isdigit()})
_MAXLEN = max(len(p) for p in _PHRASES)

ROOM_TAGS = {t for t in _P if t.startswith("R:")}
SIZE = {"A:bigger", "A:smaller"}


# --------------------------------------------------------------------------- text → tags
def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    # upper-case compass abbreviations are unambiguous ("SE"); lower-case "se" is Hindi for "from"
    for ab, full in (("NE", "northeast"), ("SE", "southeast"), ("SW", "southwest"), ("NW", "northwest")):
        t = re.sub(r"(?<![A-Za-z])" + ab[0] + r"\.?\s?" + ab[1] + r"\.?(?![A-Za-z])", full, t)
    t = t.lower()
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = re.sub(r"\b(\w+)n't\b", r"\1 not", t)              # don't → do not
    t = t.replace("don't", "do not").replace("dont ", "do not ")
    t = re.sub(r"([a-z])'s\b", r"\1s", t)                 # children's → childrens
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1\2", t)
    t = re.sub(r"\bn\.?\s*e\.?\b", "ne", t)
    return t


def _tokens(clause: str) -> List[str]:
    return [w.strip("'\"-") for w in re.findall(r"[^\s.,;:!?()\"/]+", clause) if w.strip("'\"-")]


def _repair(tokens: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Fix misspelt words against the vocabulary; returns (tokens, [(typed, fixed)])."""
    out, fixes = [], []
    known = set(_VOCAB) | {w for ph in _PHRASES for w in ph}
    for t in tokens:
        if t in known or t in _STOP or t in _NUM or len(t) < 4 or not re.search(r"[a-z]", t):
            out.append(t)
            continue
        cutoff = 0.75 if len(t) >= 6 else 0.8
        m = difflib.get_close_matches(t, _VOCAB, n=1, cutoff=cutoff)
        if m:
            out.append(m[0])
            fixes.append((t, m[0]))
        else:
            # glued words: "masterbedroom", "kitchennorth"
            split = None
            for i in range(3, len(t) - 2):
                a, b = t[:i], t[i:]
                if a in known and b in known:
                    split = [a, b]
                    break
            if split:
                out.extend(split)
            else:
                out.append(t)
    return out, fixes


def _tag(tokens: List[str]) -> List[Tuple[int, str, str]]:
    """Longest-match phrase tagging → [(token index, tag, matched text)]."""
    tags, i, n = [], 0, len(tokens)
    while i < n:
        for L in range(min(_MAXLEN, n - i), 0, -1):
            ph = tuple(tokens[i:i + L])
            if ph in _PHRASES:
                tags.append((i, _PHRASES[ph], " ".join(ph)))
                i += L
                break
        else:
            if tokens[i] in _NUM:
                tags.append((i, f"N:{_NUM[tokens[i]]}", tokens[i]))
            i += 1
    return tags


def _split_clauses(text: str) -> List[str]:
    parts = re.split(r"[.;!?\n]+|,|\bthen\b|\balso\b|\bafter that\b|\band then\b", _normalise(text))
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.extend(_split_on_and(p))
    return out


def _split_on_and(clause: str) -> List[str]:
    """Split 'make hall big and kitchen small' (two wishes) but not 'swap hall and kitchen'."""
    toks, _ = _repair(_tokens(clause))
    tags = _tag(toks)
    kinds = [t for _, t, _ in tags]
    if any(k in ("A:swap", "A:adjacent") for k in kinds):
        return [clause]
    and_pos = [i for i, t, _ in tags if t == "C:and"]
    for ap in and_pos:
        left = [t for i, t, _ in tags if i < ap]
        right = [t for i, t, _ in tags if i > ap]
        def wish(ts):
            return any(t in ROOM_TAGS or t.startswith("R:") for t in ts) and \
                any(t.startswith(("A:", "D:", "S:", "P:")) for t in ts)
        if wish(left) and wish(right):
            raw = clause.split()
            # split the raw clause at the matching 'and'-like word
            k = 0
            for idx, w in enumerate(raw):
                w2 = w.strip("'\"-,")
                if w2 in ("and", "aur", "um", "plus", "&"):
                    if k == sum(1 for i2, t2, _ in tags if t2 == "C:and" and i2 < ap):
                        return _split_on_and(" ".join(raw[:idx])) + _split_on_and(" ".join(raw[idx + 1:]))
                    k += 1
    return [clause]


# --------------------------------------------------------------------------- tags → rooms
def _room_candidates(concept: str, layout_state: Dict, owner: Optional[str] = None,
                     num: Optional[int] = None) -> List[str]:
    from floorplan_generator.intent_engine import resizable_rooms
    rooms = dict(resizable_rooms(layout_state))
    allr = layout_state.get("rooms", {})

    def by(pred):
        return [k for k, v in rooms.items() if pred(v)]

    beds = sorted(by(lambda v: v.get("category") == "BEDROOM"))
    non_master = [k for k in beds if rooms[k].get("role") != "MASTER_BEDROOM"]
    cat = {"living": "LIVING", "kitchen": "KITCHEN", "pooja": "POOJA", "study": "STUDY", "store": "STORE",
           "utility": "UTILITY", "dining": "DINING", "stairs": "STAIRS", "lift": "LIFT", "servant": "SERVANT"}
    if concept in cat:
        found = by(lambda v: v.get("category") == cat[concept] and not v.get("parent"))
        if not found and concept == "dining":                    # corridor mode: "Living & Dining"
            found = by(lambda v: v.get("category") == "LIVING")
        if not found:
            found = by(lambda v: v.get("category") == cat[concept])
        return found
    roles = {"master": "MASTER_BEDROOM", "child": "CHILD_BEDROOM", "guest": "GUEST_BEDROOM",
             "elder": "ELDER_BEDROOM"}
    if concept in roles:
        found = by(lambda v: v.get("role") == roles[concept])
        if found:
            return found
        return non_master if concept != "master" else beds[:1]
    if concept == "bedroom":
        if num:
            key = f"bed{num}"
            if key in rooms:
                return [key]
            return [beds[num - 1]] if 0 < num <= len(beds) else []
        return non_master or beds
    if concept == "bath":
        common = by(lambda v: v.get("category") == "BATHROOM" and not v.get("parent"))
        return common or by(lambda v: v.get("category") == "BATHROOM")
    if concept == "attached_bath":
        att = by(lambda v: v.get("category") == "BATHROOM" and v.get("parent"))
        if owner:
            own = [k for k in att if allr.get(k, {}).get("parent") == owner]
            if own:
                return own
        return att
    return []


def _rooms_in(tags, layout_state) -> Tuple[List[Tuple[int, List[str], str]], List[str]]:
    """[(position, candidate keys, label)] for each room mention; plus 'not in this house' notes."""
    out, missing = [], []
    used = set()
    for j, (i, t, txt) in enumerate(tags):
        if not t.startswith("R:") or j in used:
            continue
        concept, _, owner_c = t[2:].partition("@")
        owner = None
        num = None
        # neighbouring number → bedroom index ("bedroom 2", "2nd bedroom")
        for k in (j - 1, j + 1):
            if 0 <= k < len(tags) and tags[k][1].startswith("N:") and abs(tags[k][0] - i) <= 2:
                num = int(tags[k][1][2:])
        if concept == "attached_bath":
            # owner from "master bathroom" / "attached bath of master bedroom" / "master bedroom attached bath"
            if owner_c:
                c = _room_candidates(owner_c, layout_state)
                owner = c[0] if c else None
            for k in (j - 1, j + 1, j + 2):
                if 0 <= k < len(tags) and tags[k][1] in ("R:master", "R:child", "R:guest", "R:elder", "R:bedroom") \
                        and abs(tags[k][0] - i) <= 4 and owner is None:
                    c = _room_candidates(tags[k][1][2:], layout_state)
                    if c:
                        owner = c[0]
                        used.add(k)
        cands = _room_candidates(concept, layout_state, owner, num)
        label = txt
        if cands:
            out.append((i, cands, label))
        else:
            missing.append(txt)
    # drop owners that were consumed by an attached-bath mention
    keep = []
    for (i, c, lab) in out:
        if any(tags[k][0] == i for k in used):
            continue
        keep.append((i, c, lab))
    return keep, missing


# --------------------------------------------------------------------------- clause → intents
def _zones_from(tags, facing: str) -> List[str]:
    from floorplan_generator.intent_engine import cell_zone
    corners = [t[2:] for _, t, _ in tags if t.startswith("D:")]
    if corners:
        return list(dict.fromkeys(corners))
    sides = [t[2:] for _, t, _ in tags if t.startswith("S:")]
    sides = list(dict.fromkeys(sides))
    if len(sides) >= 2:
        ns = [s for s in sides if s in "NS"]
        ew = [s for s in sides if s in "EW"]
        if ns and ew:
            return [ns[0] + ew[0]]
    if sides:
        d = sides[0]
        return [z for z in ("NW", "N", "NE", "E", "SE", "S", "SW", "W") if d in z]
    plan = [t[2:] for _, t, _ in tags if t.startswith("P:")]
    if plan:
        col = -1 if "left" in plan else (1 if "right" in plan else None)
        row = 1 if "front" in plan else (-1 if "back" in plan else None)
        cols = [col] if col is not None else [-1, 0, 1]
        rows = [row] if row is not None else [1, 0, -1]
        return [cell_zone(c, r, facing) for r in rows for c in cols]
    return []


def understand_clause(clause: str, layout_state: Dict) -> Dict:
    from floorplan_generator import intent_engine as ie
    facing = layout_state.get("facing", "N")
    rooms_all = layout_state.get("rooms", {})
    toks, fixes = _repair(_tokens(clause))
    tags = _tag(toks)
    kinds = [t for _, t, _ in tags]
    acts = [t for t in kinds if t.startswith("A:")]
    rooms, missing = _rooms_in(tags, layout_state)
    res = {"clause": clause, "intents": [], "suggestions": [], "confidence": "low",
           "corrections": fixes, "note": ""}
    neg = "M:neg" in kinds
    want = "M:want" in kinds

    def nm(k):
        return rooms_all.get(k, {}).get("name", k)

    def first(c):
        return c[0] if len(c) == 1 else None

    def ambiguous(cands, make):
        res["suggestions"] = [make(k) for k in cands]
        res["note"] = "Which one? " + " / ".join(nm(k) for k in cands)

    if "U:merge" in kinds:
        res["note"] = ("Removing walls / joining two rooms into one is not possible yet — "
                       "you can ask for the rooms to be next to each other.")
        if len(rooms) >= 2 and len(rooms[0][1]) == 1 and len(rooms[1][1]) == 1 and rooms[0][1] != rooms[1][1]:
            res["suggestions"] = [{"type": "adjacent", "a": rooms[0][1][0], "b": rooms[1][1][0]}]
        return res
    if "U:site" in kinds and not rooms:
        res["note"] = ("Garden, parking, pool and other outdoor items are placed automatically by Vastu — "
                       "switch them on or off under 'Additional Components' and generate again.")
        return res

    # whole-plan mirror
    if "A:mirror" in acts or ("A:opposite" in acts and "X:plan" in kinds and not rooms):
        res["intents"] = [{"type": "mirror"}]
        res["confidence"] = "medium" if fixes else "high"
        return res

    if missing and not rooms:
        res["note"] = "Your house has no " + ", ".join(missing) + "."
        return res
    if not rooms:
        if any(a in SIZE for a in acts):
            act = "bigger" if "A:bigger" in acts else "smaller"
            f = 1.2 if act == "bigger" else 0.85
            for k in [k for k, _ in ie.resizable_rooms(layout_state)][:4]:
                res["suggestions"].append({"type": "size", "room": k, "factor": f})
            res["note"] = f"Which room should be {act}?"
        else:
            res["note"] = "Couldn't find a room in this wish."
        return res

    conf = "medium" if fixes else "high"
    r0 = rooms[0][1]

    # "no toilet near pooja" — keeping rooms apart is not an adjacency wish
    if "A:adjacent" in acts and "A:swap" not in acts and neg:
        cats = {rooms_all.get(c[0], {}).get("category") for _, c, _ in rooms if c}
        if "BATHROOM" in cats and cats & {"POOJA", "KITCHEN"}:
            res["note"] = ("Already taken care of — the planner never puts a toilet against the pooja room, "
                           "and avoids toilet walls next to the kitchen.")
            res["confidence"] = "high"
            return res
        res["note"] = "Keeping two rooms apart isn't a wish we can set yet — you can move one of them instead."
        opp = ie.opposite_intent(layout_state, r0[0]) if len(r0) == 1 else None
        if opp:
            res["suggestions"] = [opp]
        return res

    # two-room wishes
    if "A:swap" in acts or "A:adjacent" in acts:
        t = "swap" if "A:swap" in acts else "adjacent"
        if len(rooms) >= 2:
            a, b = r0, rooms[1][1]
            if len(a) == 1 and len(b) == 1 and a[0] != b[0]:
                res["intents"] = [{"type": t, "a": a[0], "b": b[0]}]
                res["confidence"] = conf
            else:
                res["suggestions"] = [{"type": t, "a": x, "b": y} for x in a for y in b if x != y][:4]
                res["note"] = "Which rooms exactly?"
        else:
            others = [k for k, v in ie.editable_rooms(layout_state)
                      if k not in r0 and v.get("group") != rooms_all.get(r0[0], {}).get("group")][:4]
            res["suggestions"] = [{"type": t, "a": r0[0], "b": o} for o in others]
            res["note"] = f"{'Swap' if t == 'swap' else 'Next to'} which room?"
        return res

    # size (with complaint inversion: "hall too small" / "kitchen is small" ⇒ bigger)
    if any(a in SIZE for a in acts):
        bigger = "A:bigger" in acts
        complaint = (("M:too" in kinds or "M:is" in kinds) and not want)
        if complaint and not bigger:
            bigger = True
            conf = "medium"
        elif complaint and bigger and "M:too" in kinds:
            bigger = False
            conf = "medium"
        f = 1.2 if bigger else 0.85
        if first(r0):
            res["intents"] = [{"type": "size", "room": r0[0], "factor": f}]
            res["confidence"] = conf
        else:
            ambiguous(r0, lambda k: {"type": "size", "room": k, "factor": f})
        return res

    zones = _zones_from(tags, facing)
    # "other side"
    if "A:opposite" in acts and not zones:
        if first(r0):
            it = ie.opposite_intent(layout_state, r0[0])
            if it:
                res["intents"] = [it]
                res["confidence"] = conf
            else:
                res["note"] = f"No reachable 'other side' for {nm(r0[0])} in this layout."
        else:
            ambiguous(r0, lambda k: ie.opposite_intent(layout_state, k) or {"type": "lock", "room": k})
        return res

    # direction (with negation: "no kitchen in north east" ⇒ anywhere else reachable)
    if zones:
        def zone_it(k):
            if neg:
                ok = [z for z in ie.feasible_zones(layout_state, k) if z not in zones]
                return {"type": "zone", "room": k, "zones": ok, "avoid": zones}
            return {"type": "zone", "room": k, "zones": zones}
        if first(r0):
            res["intents"] = [zone_it(r0[0])]
            res["confidence"] = conf
        else:
            ambiguous(r0, zone_it)
        return res

    if "A:lock" in acts:
        if first(r0):
            res["intents"] = [{"type": "lock", "room": r0[0]}]
            res["confidence"] = conf
        else:
            ambiguous(r0, lambda k: {"type": "lock", "room": k})
        return res

    # a room with no clear action → offer the likely wishes
    k = r0[0]
    sugg = []
    best = best_zone(layout_state, k)
    if best and best != ie.current_zone(layout_state, k):
        sugg.append({"type": "zone", "room": k, "zones": [best]})
    sugg.append({"type": "size", "room": k, "factor": 1.2})
    opp = ie.opposite_intent(layout_state, k)
    if opp:
        sugg.append(opp)
    res["suggestions"] = sugg
    res["note"] = f"What should change in the {nm(k)}?"
    return res


def best_zone(layout_state: Dict, key: str) -> Optional[str]:
    """Best-scoring reachable Vastu zone for a room (None if nothing reachable)."""
    from floorplan_generator import intent_engine as ie
    zs = ie.feasible_zones(layout_state, key)
    if not zs:
        return None
    return max(zs, key=lambda z: (ie.zone_fit(layout_state, key, z)[0], -("CENTER" == z)))


def understand(text: str, layout_state: Dict) -> List[Dict]:
    """Every clause of the text, understood."""
    return [understand_clause(c, layout_state) for c in _split_clauses(text)]


def suggest_wishes(layout_state: Dict, limit: int = 4) -> List[Dict]:
    """Ready-made wishes for this house: Vastu fixes for key rooms, a bigger living room, a bath flip."""
    from floorplan_generator import intent_engine as ie
    rooms = layout_state.get("rooms", {})
    out: List[Dict] = []
    for k, v in ie.editable_rooms(layout_state):
        if v.get("role") == "MASTER_BEDROOM" or v.get("category") in ("KITCHEN", "POOJA"):
            b = best_zone(layout_state, k)
            if b and b != ie.current_zone(layout_state, k):
                out.append({"type": "zone", "room": k, "zones": [b], "why": "best Vastu"})
    liv = [k for k, v in rooms.items() if v.get("category") == "LIVING"]
    if liv:
        out.append({"type": "size", "room": liv[0], "factor": 1.2})
    att = [k for k, v in rooms.items() if v.get("category") == "BATHROOM" and v.get("parent")]
    if att:
        it = ie.opposite_intent(layout_state, att[0])
        if it:
            out.append(it)
    kit = [k for k, v in rooms.items() if v.get("category") == "KITCHEN"]
    if kit:
        out.append({"type": "size", "room": kit[0], "factor": 1.2})
    return out[:limit]

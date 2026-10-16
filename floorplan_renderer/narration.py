"""
🗣️ The guide's words — the guided walk told the way a good broker shows a home
==============================================================================
Every stop of the 3D guided walk (floorplan_renderer/walkthrough.py) gets its words in three languages:

  en  simple Indian English — short sentences, everyday words ("hall", "car parking is also there"),
      the English an owner who learned it at school speaks and understands
  te  Telugu as it is spoken around Guntur — colloquial (వస్తది, ఉంటది, అండీ), with the English words
      brokers really use (హాల్, కిచెన్, ఈస్ట్ ఫేసింగ్) and Vastu's own directions (ఈశాన్యం, ఆగ్నేయం…)
  hi  natural spoken Hindi — जी, देखिए, एकदम; Vastu's directions (ईशान कोण, आग्नेय कोण…)

Like a good broker it shows a home in its best light — the morning sun, the Vastu corner, the king-size bed,
the bathroom of your own, the rent from the other floors — and only the good: every claim is measured from the
plan (sizes, Vastu zones and scores, windows, furniture, what opens into what, gates, stairs, lift) and a room
only gets the good things that are true of it; nothing weak is ever said (room_highlights).

Each line is made twice from the same template: the caption (digits, short units) and the speech
(numbers in words, no symbols) as a list of sentences. A sentence may carry a cue (C(name, text)): what the
camera looks at while it is said — the bed, the East window, the gate, the sump, the South-West corner of the
roof. walkthrough._cues measures each cue's point for the stop; add_talk puts it on the spoken line as
'at' = [x, y, z, g] (g = 1: z above the ground, 0: above the floor the guide stands on) and walk.js turns
(and gently zooms) the camera to it the moment that sentence starts. A sentence's key k = sha1(lang|text)[:16] names its
voice clip: model_folder/voice_engine.py has them spoken offline, walk.js plays voice/<voice>/<k>.wav and
falls back to the browser's own voice for a clip that is not ready.

Pure Python, no network, no model: the words are written here, not generated.
"""

import hashlib
import re
from typing import Dict, List, Optional

LANGS = ("en", "te", "hi")
LANG_NAMES = {"en": "English", "te": "తెలుగు", "hi": "हिन्दी"}
SQFT = 10.7639
M2_PER_CENT = 40.4686
GAJ = 1.19599                           # square yards per m²


# =============================================================================
# Numbers in words
# =============================================================================

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
            "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def en_words(n: int) -> str:
    n = int(n)
    if n < 0:
        return "minus " + en_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + _EN_ONES[n % 10])
    if n < 1000:
        return _EN_ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + en_words(n % 100))
    if n < 100000:
        return en_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + en_words(n % 1000))
    if n < 10 ** 7:
        return en_words(n // 100000) + " lakh" + ("" if n % 100000 == 0 else " " + en_words(n % 100000))
    return en_words(n // 10 ** 7) + " crore" + ("" if n % 10 ** 7 == 0 else " " + en_words(n % 10 ** 7))


_TE_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
            "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TE_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]


def te_words(n: int, attr: bool = False) -> str:
    """Telugu number; attr=True before a noun (ఒక గది, రెండు వందల అడుగులు)."""
    n = int(n)
    if n < 20:
        return "ఒక" if (attr and n == 1) else _TE_ONES[n]
    if n < 100:
        return _TE_TENS[n // 10] + ("" if n % 10 == 0 else " " + te_words(n % 10, attr))
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1:
            head = "వంద" if r == 0 else "నూట"
        else:
            head = te_words(h, True) + (" వందలు" if r == 0 and not attr else " వందల")
        return head if r == 0 else head + " " + te_words(r, attr)
    if n < 100000:
        t, r = divmod(n, 1000)
        head = "వెయ్యి" if t == 1 else te_words(t, True) + (" వేలు" if r == 0 and not attr else " వేల")
        return head if r == 0 else head + " " + te_words(r, attr)
    if n < 10 ** 7:
        l, r = divmod(n, 100000)
        head = "ఒక లక్ష" if l == 1 else te_words(l, True) + (" లక్షలు" if r == 0 and not attr else " లక్షల")
        return head if r == 0 else head + " " + te_words(r, attr)
    c, r = divmod(n, 10 ** 7)
    head = "ఒక కోటి" if c == 1 else te_words(c, True) + " కోట్ల"
    return head if r == 0 else head + " " + te_words(r, attr)


_HI = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
       "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
       "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
       "इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ "
       "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
       "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे "
       "सत्तानवे अट्ठानवे निन्यानवे").split()


def hi_words(n: int) -> str:
    n = int(n)
    if n < 100:
        return _HI[n]
    if n < 1000:
        h, r = divmod(n, 100)
        head = "सौ" if h == 1 and r == 0 else _HI[h] + " सौ"
        return head if r == 0 else head + " " + _HI[r]
    if n < 100000:
        t, r = divmod(n, 1000)
        return hi_words(t) + " हज़ार" + ("" if r == 0 else " " + hi_words(r))
    if n < 10 ** 7:
        l, r = divmod(n, 100000)
        return hi_words(l) + " लाख" + ("" if r == 0 else " " + hi_words(r))
    c, r = divmod(n, 10 ** 7)
    return hi_words(c) + " करोड़" + ("" if r == 0 else " " + hi_words(r))


_DIGIT = {"en": "zero one two three four five six seven eight nine".split(),
          "te": "జీరో వన్ టూ త్రీ ఫోర్ ఫైవ్ సిక్స్ సెవెన్ ఎయిట్ నైన్".split(),
          "hi": "ज़ीरो वन टू थ्री फ़ोर फ़ाइव सिक्स सेवन एट नाइन".split()}
_LETTER = {"en": {"G": "G"}, "te": {"G": "జీ"}, "hi": {"G": "जी"}}
_ORD_EN = ["ground", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
           "eleventh", "twelfth"]
_ORD_TE = ["గ్రౌండ్", "ఫస్ట్", "సెకండ్", "థర్డ్", "ఫోర్త్", "ఫిఫ్త్", "సిక్స్త్", "సెవెంత్", "ఎయిత్", "నైన్త్", "టెన్త్",
           "ఎలెవెన్త్", "ట్వెల్ఫ్త్"]
_ORD_HI = ["ग्राउंड", "फ़र्स्ट", "सेकंड", "थर्ड", "फ़ोर्थ", "फ़िफ़्थ", "सिक्स्थ", "सेवंथ", "एटथ", "नाइंथ", "टेंथ",
           "इलेवंथ", "ट्वेल्फ़्थ"]
_PADA_ORD = {"en": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"],
             "te": ["మొదటి", "రెండో", "మూడో", "నాలుగో", "ఐదో", "ఆరో", "ఏడో", "ఎనిమిదో"],
             "hi": ["पहले", "दूसरे", "तीसरे", "चौथे", "पाँचवें", "छठे", "सातवें", "आठवें"]}   # oblique: "तीसरे पद में"


def cap(s: str) -> str:
    """First letter up, the rest untouched ('even a big SUV' → 'Even a big SUV')."""
    return s[:1].upper() + s[1:]


class _Said(str):
    """A sentence that knows what the camera should look at while it is said."""
    cue: Optional[str] = None


def C(cue: Optional[str], text: Optional[str]) -> Optional[str]:
    """Tag a sentence with a cue name (resolved to a point per stop by walkthrough._cues)."""
    if text is None:
        return None
    out = _Said(text)
    out.cue = cue
    return out


class T:
    """
    One language in one mode: `speech=False` writes captions (digits), `speech=True` writes what is said
    (numbers in words). Templates call T.p(en, te, hi) with the three sentences side by side.
    """

    def __init__(self, lang: str, speech: bool):
        self.lang, self.speech = lang, speech

    def p(self, en: str, te: str, hi: str) -> str:
        return {"en": en, "te": te, "hi": hi}[self.lang]

    # numbers ------------------------------------------------------------------
    def i(self, v, attr: bool = True) -> str:
        """A whole number (rounded)."""
        n = int(round(float(v)))
        if not self.speech:
            return f"{n:,}"
        return {"en": en_words, "te": lambda k: te_words(k, attr), "hi": hi_words}[self.lang](n)

    def h(self, v) -> str:
        """To the nearest half: 4.5 → 'four and a half' / 'నాలుగున్నర' / 'साढ़े चार'."""
        x = round(float(v) * 2) / 2
        n, half = int(x), (x - int(x)) >= 0.5
        if not self.speech:
            return f"{n}" if not half else f"{x:.1f}"
        if not half:
            return self.i(n)
        if self.lang == "en":
            return "half" if n == 0 else f"{en_words(n)} and a half"
        if self.lang == "te":
            if n == 0:
                return "అర"
            return te_words(n) + "న్నర" if n < 20 else te_words(n, True)
        return {0: "आधा", 1: "डेढ़", 2: "ढाई"}.get(n) or "साढ़े " + hi_words(n)

    def ft(self, m: float) -> str:
        return self.i(m / 0.3048)

    def inch(self, mm: float) -> str:
        return self.h(mm / 25.4)

    def digits(self, s: str) -> str:
        """A flat number read the way people say it: 301 → 'three zero one'."""
        if not self.speech:
            return s
        out = []
        for ch in s:
            if ch.isdigit():
                out.append(_DIGIT[self.lang][int(ch)])
            elif ch.upper() in _LETTER[self.lang]:
                out.append(_LETTER[self.lang][ch.upper()])
        return " ".join(out)

    def bhk(self, s: str) -> str:
        m = re.search(r"(\d)", s or "")
        if not m:
            return s or ""
        n = int(m.group(1))
        if not self.speech:
            return f"{n}BHK"
        return self.p(f"{en_words(n)} B H K", f"{_DIGIT['te'][n]} బీహెచ్కే", f"{_DIGIT['hi'][n]} बीएचके")

    def floor(self, L: int, stilt_level: bool = False) -> str:
        if stilt_level:
            return self.p("the parking floor", "పార్కింగ్ ఫ్లోర్", "पार्किंग फ़्लोर")
        k = max(0, min(int(L), len(_ORD_EN) - 1))
        return self.p(f"the {_ORD_EN[k]} floor", f"{_ORD_TE[k]} ఫ్లోర్", f"{_ORD_HI[k]} फ़्लोर")

    def join(self, words: List[str]) -> str:
        words = [w for w in words if w]
        if len(words) <= 1:
            return "".join(words)
        last = self.p(" and ", ", ", " और ")
        return ", ".join(words[:-1]) + last + words[-1]


# =============================================================================
# Words for places, rooms and directions
# =============================================================================

FACING = {"en": {"N": "North", "E": "East", "S": "South", "W": "West", "NE": "North-East", "NW": "North-West",
                 "SE": "South-East", "SW": "South-West"},
          "te": {"N": "నార్త్", "E": "ఈస్ట్", "S": "సౌత్", "W": "వెస్ట్", "NE": "నార్త్ ఈస్ట్", "NW": "నార్త్ వెస్ట్",
                 "SE": "సౌత్ ఈస్ట్", "SW": "సౌత్ వెస్ట్"},
          "hi": {"N": "नॉर्थ", "E": "ईस्ट", "S": "साउथ", "W": "वेस्ट", "NE": "नॉर्थ ईस्ट", "NW": "नॉर्थ वेस्ट",
                 "SE": "साउथ ईस्ट", "SW": "साउथ वेस्ट"}}
DIR = {"en": {"N": "North", "NE": "North-East", "E": "East", "SE": "South-East", "S": "South", "SW": "South-West",
              "W": "West", "NW": "North-West", "CENTER": "centre"},
       "te": {"N": "ఉత్తరం", "NE": "ఈశాన్యం", "E": "తూర్పు", "SE": "ఆగ్నేయం", "S": "దక్షిణం", "SW": "నైరుతి",
              "W": "పడమర", "NW": "వాయువ్యం", "CENTER": "ఇంటి మధ్య భాగం"},
       "hi": {"N": "उत्तर", "NE": "ईशान कोण", "E": "पूर्व", "SE": "आग्नेय कोण", "S": "दक्षिण", "SW": "नैऋत्य कोण",
              "W": "पश्चिम", "NW": "वायव्य कोण", "CENTER": "घर का बीच का हिस्सा"}}
LOC = {"en": {"N": "in the North", "NE": "in the North-East corner", "E": "in the East", "SE": "in the South-East corner",
              "S": "in the South", "SW": "in the South-West corner", "W": "in the West", "NW": "in the North-West corner",
              "CENTER": "in the centre of the house"},
       "te": {"N": "ఉత్తరం వైపు", "NE": "ఈశాన్యంలో", "E": "తూర్పు వైపు", "SE": "ఆగ్నేయంలో", "S": "దక్షిణం వైపు",
              "SW": "నైరుతిలో", "W": "పడమర వైపు", "NW": "వాయువ్యంలో", "CENTER": "ఇంటి మధ్యలో"},
       "hi": {"N": "उत्तर में", "NE": "ईशान कोण में", "E": "पूर्व में", "SE": "आग्नेय कोण में", "S": "दक्षिण में",
              "SW": "नैऋत्य कोण में", "W": "पश्चिम में", "NW": "वायव्य कोण में", "CENTER": "घर के बीच में"}}

# (en, te, hi) by the room's kind (see room_kind)
ROOM = {
    "LIVING": ("hall", "హాల్", "हॉल"),
    "FAMILY": ("family lounge", "ఫ్యామిలీ లాంజ్", "फ़ैमिली लाउंज"),
    "UPPER_HALL": ("upstairs hall", "పై హాల్", "ऊपर का हॉल"),
    "DINING": ("dining hall", "డైనింగ్ హాల్", "डाइनिंग हॉल"),
    "KITCHEN": ("kitchen", "కిచెన్", "किचन"),
    "MASTER_BEDROOM": ("master bedroom", "మాస్టర్ బెడ్రూమ్", "मास्टर बेडरूम"),
    "CHILD_BEDROOM": ("children's bedroom", "పిల్లల బెడ్రూమ్", "बच्चों का बेडरूम"),
    "ELDER_BEDROOM": ("elders' bedroom", "పెద్దవాళ్ళ బెడ్రూమ్", "बड़ों का बेडरूम"),
    "GUEST_BEDROOM": ("guest bedroom", "గెస్ట్ బెడ్రూమ్", "गेस्ट बेडरूम"),
    "BEDROOM": ("bedroom", "బెడ్రూమ్", "बेडरूम"),
    "ATTACHED_BATH": ("attached bathroom", "అటాచ్డ్ బాత్రూమ్", "अटैच्ड बाथरूम"),
    "BATHROOM": ("common bathroom", "కామన్ బాత్రూమ్", "कॉमन बाथरूम"),
    "POOJA": ("pooja room", "పూజ గది", "पूजा घर"),
    "STUDY": ("study room", "స్టడీ రూమ్", "स्टडी रूम"),
    "STORE": ("store room", "స్టోర్ రూమ్", "स्टोर रूम"),
    "UTILITY": ("utility area", "యుటిలిటీ", "यूटिलिटी"),
    "FOYER": ("entrance foyer", "ఎంట్రన్స్ ఫోయర్", "एंट्रेंस फ़ॉयर"),
    "BALCONY": ("balcony", "బాల్కనీ", "बालकनी"),
    "SERVANT": ("servant room", "సర్వెంట్ రూమ్", "सर्वेंट रूम"),
    "CORRIDOR": ("passage", "పాసేజ్", "पैसेज"),
    "LOBBY": ("lobby", "లాబీ", "लॉबी"),
    "DRESS": ("dressing room", "డ్రెస్సింగ్ రూమ్", "ड्रेसिंग रूम"),
    "ROOM": ("room", "గది", "कमरा"),
}

# what the main door's pada is said to bring, for the auspicious ones (Vastu's traditional readings)
PADA_GIFT = {
    "N3": ("wealth", "సంపద", "धन-दौलत"),
    "N4": ("prosperity", "అభివృద్ధి", "तरक़्क़ी"),
    "N5": ("peace and steady money", "ప్రశాంతత, ఆర్థిక స్థిరత్వం", "शांति और पैसे में स्थिरता"),
    "E3": ("success", "విజయం", "कामयाबी"),
    "E4": ("name and respect", "పేరు, హోదా", "मान-सम्मान"),
    "S3": ("gains in business", "వ్యాపారంలో లాభం", "कारोबार में फ़ायदा"),
    "S4": ("a growing family", "కుటుంబ అభివృద్ధి", "परिवार की खुशहाली"),
    "W3": ("growth in studies and business", "చదువు, వ్యాపార వృద్ధి", "पढ़ाई और कारोबार में बढ़त"),
    "W4": ("happiness of children and goodwill", "సంతాన సుఖం, మంచి పేరు", "संतान सुख और अच्छा नाम"),
}
PADA_NAME = {
    "N3": ("Mukhya", "ముఖ్య", "मुख्य"), "N4": ("Bhallata", "భల్లాట", "भल्लाट"), "N5": ("Kubera", "కుబేర", "कुबेर"),
    "N7": ("Aditi", "అదితి", "अदिति"), "E2": ("Parjanya", "పర్జన్య", "पर्जन्य"), "E3": ("Jayanta", "జయంత", "जयंत"),
    "E4": ("Indra", "ఇంద్ర", "इंद्र"), "S3": ("Vitatha", "వితథ", "वितथ"), "S4": ("Brihatkshata", "బృహత్క్షత", "बृहत्क्षत"),
    "W3": ("Sugriva", "సుగ్రీవ", "सुग्रीव"), "W4": ("Pushpadanta", "పుష్పదంత", "पुष्पदंत"), "W5": ("Varuna", "వరుణ", "वरुण"),
}


def _tr(t: T, triple) -> str:
    return triple[LANGS.index(t.lang)]


def room_kind(name: str, cat: str, role: str = "", attached: bool = False) -> str:
    """The kind of room a broker names (master bedroom, attached bathroom, family lounge…)."""
    n = (name or "").split(" · ")[-1].lower()
    if cat == "BEDROOM":
        if role in ROOM:
            return role
        for key, word in (("MASTER_BEDROOM", "master"), ("ELDER_BEDROOM", "elder"), ("CHILD_BEDROOM", "child"),
                          ("GUEST_BEDROOM", "guest")):
            if word in n:
                return key
        return "BEDROOM"
    if cat == "BATHROOM":
        return "ATTACHED_BATH" if (attached or "attached" in n) else "BATHROOM"
    if "lobby" in n or "gallery" in n or role == "COMMON":
        return "LOBBY"
    if "dress" in n or "wardrobe" in n:
        return "DRESS"
    if cat == "LIVING":
        if "lounge" in n:
            return "FAMILY"
        if "upper" in n:
            return "UPPER_HALL"
        return "LIVING"
    return cat if cat in ROOM else "ROOM"


# =============================================================================
# Templates — each returns sentences; the same code writes captions and speech
# =============================================================================

_BED = {"king": ("king size", "కింగ్ సైజ్", "किंग साइज़"), "queen": ("queen size", "క్వీన్ సైజ్", "क्वीन साइज़"),
        "double": ("double", "డబుల్", "डबल"), "single": ("single", "సింగిల్", "सिंगल")}


# =============================================================================
# The highlights — only what is good about a room, measured, best first
# =============================================================================
# A room's highlights come from its facts (size, Vastu zone and score, windows by compass side, the furniture
# that fits, the rooms it opens into, the walls it shares, how far it sits from the road). Each candidate has a
# weight; the strongest few are said in the order a broker shows a room — welcome, the wow, Vastu, light, what
# fits, how it lives — and a bedroom with its own bathroom ends by leading you into it. Nothing weak is ever
# said: a room only gets the good things that are true of it. Across the house, a highlight said before weighs
# less (morning sun three times sounds like a script), and every highlight has more than one wording, picked
# by the room's place in the walk, so no two rooms sound alike.

HABITABLE = {"LIVING", "FAMILY", "UPPER_HALL", "DINING", "KITCHEN", "STUDY", "MASTER_BEDROOM", "CHILD_BEDROOM",
             "ELDER_BEDROOM", "GUEST_BEDROOM", "BEDROOM"}
WET = {"BATHROOM", "ATTACHED_BATH"}
_REPEATS = {"east", "north", "south", "west", "cross", "quiet", "roomy", "v_best", "v_good", "night", "pipes",
            "bath_window", "vent", "fan", "inner", "v_bath"}
_FLOW = {"big": 1, "vastu": 2, "light": 3, "fit": 4, "life": 5}
_LIMIT = {"fit": 2, "life": 2}


def room_highlights(f: Dict, used: Optional[Dict[str, int]] = None) -> List[List]:
    """The good things to say about a room, strongest first, then in speaking order: [[id, args], …]."""
    kind = f.get("kind", "ROOM")
    used = used if used is not None else {}
    bed = kind.endswith("BEDROOM")
    wet = kind in WET
    area, z, sc = float(f.get("area") or 0), f.get("zone"), f.get("score")
    wins = list(dict.fromkeys(f.get("wins") or []))
    fits = f.get("fits") or {}
    links, mates = set(f.get("links") or []), set(f.get("mates") or [])
    cand = []

    def add(w, group, pid, **a):
        cand.append((w - (2.5 * used.get(pid, 0) if pid in _REPEATS else 0.0), group, pid, a))
    # the wow: size
    if bed and f.get("largest"):
        add(7, "big", "largest")
    elif (kind == "LIVING" and area >= 14) or (kind in ("FAMILY", "UPPER_HALL") and area >= 12) or \
            (bed and area >= 12.5) or (kind == "KITCHEN" and area >= 8.5) or (kind == "DINING" and area >= 10):
        add(5, "big", "roomy")
    elif wet and area >= 3.6:
        add(3.5, "big", "bath_roomy")
    # Vastu, only where it is good
    if sc is not None and z:
        special = (kind == "POOJA" and z == "NE") or (kind == "KITCHEN" and z in ("SE", "NW")) or \
                  (kind == "MASTER_BEDROOM" and z == "SW") or (kind in ("LIVING", "FAMILY") and z in ("N", "NE", "E") and sc >= 3)
        if special:
            add(9, "vastu", "v_special")
        elif wet:
            if sc >= 5:
                add(4, "vastu", "v_bath")
        elif kind in ("STORE", "UTILITY", "CORRIDOR", "LOBBY", "FOYER", "BALCONY", "DRESS"):
            pass
        elif sc >= 8:
            add(6, "vastu", "v_best")
        elif sc >= 5:
            add(5, "vastu", "v_good")
    # light and air
    if wet:
        if wins:
            add(4, "light", "bath_window")
        elif f.get("vent"):
            add(3, "light", "vent")
        else:
            add(1.5, "light", "fan")
    elif kind not in ("CORRIDOR", "LOBBY", "STORE", "DRESS", "FOYER"):
        if len(wins) >= 2:
            add(7, "light", "cross", sides=wins[:2])
        elif wins == ["E"]:
            add(6, "light", "east")
        elif wins == ["N"]:
            add(5 + (1 if kind in ("STUDY", "CHILD_BEDROOM") else 0), "light", "north")
        elif wins == ["S"]:
            add(3, "light", "south")
        elif wins == ["W"]:
            add(3, "light", "west")
        elif not wins and kind in HABITABLE:
            add(1.5, "light", "inner")
    # what fits
    if fits.get("bed"):
        add(6 if fits["bed"] in ("king", "queen") else 4.5, "fit", "bed", bed=fits["bed"],
            wardrobe=bool(fits.get("wardrobe")))
    if fits.get("desk") and kind != "STUDY":
        add(4 + (1 if kind == "CHILD_BEDROOM" else 0), "fit", "desk")
    if fits.get("sofa"):
        add(5, "fit", "sofa", n=int(fits["sofa"]))
    if fits.get("dining"):
        add(5, "fit", "dining", n=int(fits["dining"]))
    if fits.get("counter"):
        add(5, "fit", "counter", m=float(fits["counter"]))
    if fits.get("washer"):
        add(3, "fit", "washer")
    if wet and (fits.get("wc") or fits.get("shower")):
        add(4, "fit", "fixtures", wc=bool(fits.get("wc")), shower=bool(fits.get("shower")),
            basin=bool(fits.get("basin")))
    # how it lives: what it opens into, where it sits
    if kind == "KITCHEN" and links & {"DINING", "LIVING", "FAMILY"}:
        add(6, "life", "serve")
    if kind == "KITCHEN" and "UTILITY" in links:
        add(4, "life", "utility")
    if kind == "UTILITY" and "KITCHEN" in links:
        add(4, "life", "utility_back")
    if kind == "DINING" and "KITCHEN" in links:
        add(5, "life", "dining_kitchen")
    if kind == "LIVING" and "FOYER" in links:
        add(3, "life", "privacy")
    if kind == "POOJA" and links & {"LIVING", "FAMILY", "FOYER", "DINING"}:
        add(4, "life", "pooja_hall")
    if bed and float(f.get("depth") or 0) >= 0.6:
        add(4, "life", "quiet")
    if kind == "CHILD_BEDROOM" and "MASTER_BEDROOM" in (links | mates):
        add(3.5, "life", "near_parents")
    if kind == "ELDER_BEDROOM":
        if f.get("ground"):
            add(7, "life", "elder_ground")
        elif f.get("lift"):
            add(5, "life", "elder_lift")
    if wet and f.get("parent"):
        add(5, "life", "night")
        if f.get("parent") == "ELDER_BEDROOM":
            add(4.5, "life", "elder_safe")
    if kind == "BATHROOM" and not f.get("parent") and links & {"LIVING", "DINING", "CORRIDOR", "FOYER", "FAMILY",
                                                               "UPPER_HALL", "LOBBY"}:
        add(3, "life", "guests")
    if wet and f.get("wet_share"):
        add(3.5, "life", "pipes")
    if kind == "DRESS":
        add(5, "life", "dress")
    # the strongest few, at most one of a kind (two things that fit, two about living), in speaking order
    keep, per = [], {}
    k_max = 3 if (wet or kind in ("DRESS", "CORRIDOR", "STORE", "UTILITY", "FOYER", "POOJA", "LOBBY")) else 4
    for w, group, pid, a in sorted(cand, key=lambda c: -c[0]):
        if w <= 0 or per.get(group, 0) >= _LIMIT.get(group, 1):
            continue
        per[group] = per.get(group, 0) + 1
        keep.append((group, pid, a))
        if len(keep) >= k_max:
            break
    keep.sort(key=lambda g: _FLOW[g[0]])
    return [[pid, a] for _, pid, a in keep]


def _pick(t: T, f: Dict, pid: str, a: Optional[Dict], variants) -> str:
    """
    One of a highlight's wordings: the next one each time the walk says it again (the ledger counts, '_n'),
    else by the room's place in the walk — so the same words never come twice in a row.
    """
    n = (a or {}).get("_n")
    k = (n if n is not None else int(f.get("seq") or 0) + sum(map(ord, pid))) % len(variants)
    return t.p(*variants[k])


def _size(t: T, f: Dict) -> str:
    w, h, sqft = f.get("w", 0), f.get("h", 0), f.get("sqft", 0)
    return t.p(f"{t.ft(w)} by {t.ft(h)} feet, {t.i(sqft)} square feet",
               f"{t.ft(w)} బై {t.ft(h)} అడుగులు, {t.i(sqft)} స్క్వేర్ ఫీట్",
               f"{t.ft(w)} बाई {t.ft(h)} फ़ीट, {t.i(sqft)} स्क्वेयर फ़ीट")


def _intro(t: T, f: Dict, c: Dict) -> str:
    """Welcome into the room: what it is and how big."""
    kind = f["kind"]
    name = _tr(t, ROOM.get(kind, ROOM["ROOM"]))
    size = _size(t, f)
    if kind in ("LIVING", "FAMILY", "UPPER_HALL"):
        return t.p(f"See, this is the {name} — {size}.", f"చూడండి, ఇదే {name} — {size}.", f"देखिए, ये रहा {name} — {size}।")
    if kind == "MASTER_BEDROOM":
        return t.p(f"This is the master bedroom, for you both — {size}.", f"ఇదే మాస్టర్ బెడ్రూమ్ అండీ — మీ ఇద్దరి కోసం. {size}.",
                   f"ये मास्टर बेडरूम है — आप दोनों के लिए। {size}।")
    if kind == "CHILD_BEDROOM":
        return t.p(f"This one is for the children — {size}.", f"ఇది పిల్లల గది — {size}.", f"ये बच्चों का कमरा है — {size}।")
    if kind == "ELDER_BEDROOM":
        return t.p(f"This is for the elders — your parents. {size}.", f"ఇది పెద్దవాళ్ళ గది — అమ్మా నాన్నల కోసం. {size}.",
                   f"ये बड़ों का कमरा है — माता-पिता के लिए। {size}।")
    if kind == "GUEST_BEDROOM":
        return t.p(f"When relatives come, this guest room is theirs — {size}. Privacy for them, privacy for you.",
                   f"చుట్టాలు వస్తే ఈ గెస్ట్ బెడ్రూమ్ వాళ్ళదే — {size}. వాళ్ళకీ ప్రైవసీ, మీకూ ప్రైవసీ.",
                   f"मेहमान आएँ तो ये गेस्ट रूम उनका — {size}। उन्हें भी प्राइवेसी, आपको भी।")
    if kind == "KITCHEN":
        return t.p(f"Now the kitchen — {size}.", f"ఇక కిచెన్ చూద్దాం — {size}.", f"अब किचन देखिए — {size}।")
    if kind == "POOJA":
        return t.p("This is the pooja room — a separate place for God.", "ఇదే పూజ గది — దేవుడికి సెపరేట్ గా ఒక గది.",
                   "ये पूजा घर है — भगवान के लिए अलग जगह।")
    if kind == "STUDY":
        return t.p(f"Study room — {size}. For work from home or the children's studies, a quiet place.",
                   f"స్టడీ రూమ్ — {size}. వర్క్ ఫ్రమ్ హోమ్ కి, పిల్లల చదువుకి — ప్రశాంతంగా ఉంటది.",
                   f"स्टडी रूम — {size}। वर्क फ़्रॉम होम हो या बच्चों की पढ़ाई — शांत जगह।")
    if kind == "STORE":
        return t.p("Store room — rice bags, old things, everything goes in here.",
                   "స్టోర్ రూమ్ — బియ్యం బస్తాలు, పాత సామాన్లు అన్నీ ఇక్కడే పెట్టుకోవచ్చు.",
                   "स्टोर रूम — अनाज की बोरियाँ, पुराना सामान, सब यहीं।")
    if kind == "UTILITY":
        return t.p("Utility area — a separate place for washing clothes and vessels.",
                   "యుటిలిటీ — బట్టలు ఉతకడానికి, అంట్లు తోముకోవడానికి సెపరేట్ ప్లేస్.",
                   "यूटिलिटी — कपड़े धोने और बर्तन माँजने की अलग जगह।")
    if kind == "FOYER":
        return t.p("Entrance foyer — keep the shoe stand here; from outside, nobody sees the whole house.",
                   "ఎంట్రన్స్ ఫోయర్ — చెప్పుల స్టాండ్ ఇక్కడే; బయటి వాళ్ళకి ఇల్లంతా కనిపించదు.",
                   "एंट्रेंस फ़ॉयर — जूतों का स्टैंड यहीं; बाहर वालों को पूरा घर नहीं दिखता।")
    if kind == "BALCONY":
        return t.p("Balcony — for evening tea and fresh air.", "బాల్కనీ — సాయంత్రం టీ తాగుతూ కూర్చోవడానికి.",
                   "बालकनी — शाम की चाय और ताज़ी हवा के लिए।")
    if kind == "SERVANT":
        return t.p("Servant room — a separate room for the helper.", "సర్వెంట్ రూమ్ — పని వాళ్ళకి సెపరేట్ గది.",
                   "सर्वेंट रूम — काम वालों के लिए अलग कमरा।")
    if kind == "DRESS":
        return t.p(f"This is the dressing room — {size}.", f"ఇదే డ్రెస్సింగ్ రూమ్ — {size}.", f"ये ड्रेसिंग रूम है — {size}।")
    if kind == "ATTACHED_BATH" and f.get("parent"):
        par = _tr(t, ROOM.get(f["parent"], ROOM["BEDROOM"]))
        if t.lang == "hi":
            par = par.replace("का ", "के ")                  # बड़ों के बेडरूम का अपना बाथरूम
        return t.p(f"And here is the {par}'s own bathroom — straight from the room.",
                   f"ఇదే {par} కి అటాచ్డ్ బాత్రూమ్ అండీ — గదిలో నుంచే నేరుగా.",
                   f"और ये रहा {par} का अपना बाथरूम — कमरे से सीधे।")
    if kind in WET:
        return t.p(f"This is the {name}.", f"ఇది {name}.", f"ये {name} है।")
    if kind == "DINING":
        return t.p(f"Dining hall — {size}.", f"డైనింగ్ హాల్ — {size}.", f"डाइनिंग हॉल — {size}।")
    if kind in ("CORRIDOR", "LOBBY"):
        if len(f.get("links") or []) >= 3:
            return t.p("Every room opens off this passage — nobody walks through anybody's room.",
                       "ప్రతి గదికీ ఈ పాసేజ్ నుంచే దారి — ఒకరి గదిలోంచి ఇంకొకరు నడవాల్సిన పనే లేదు.",
                       "हर कमरे का रास्ता इसी पैसेज से — किसी को किसी के कमरे से होकर नहीं जाना पड़ता।")
        return _pick(t, f, "passage", None, [("This way, please.", "ఇటు రండి.", "इधर आइए।"),
                                       ("Come, this way.", "రండి, ఇటు వైపు.", "आइए, इस तरफ़।")])
    return t.p(f"This is the {name} — {size}.", f"ఇది {name} — {size}.", f"ये {name} है — {size}।")


def _special_vastu(t: T, kind: str, f: Dict) -> str:
    z = f.get("zone")
    loc = LOC[t.lang].get(z, "")
    if kind == "POOJA":
        return t.p("North-East is God's corner in Vastu — there is no better place for pooja.",
                   "ఈశాన్యం దేవుడి మూల — పూజకి అంతకంటే మంచి ప్లేస్ లేదు అని వాస్తు చెప్తది.",
                   "ईशान कोण भगवान का कोना है — पूजा के लिए इससे अच्छी जगह नहीं।")
    if kind == "KITCHEN" and z == "SE":
        return t.p("South-East is the fire corner — Vastu says cooking belongs here. A proper Vastu kitchen.",
                   "ఆగ్నేయం అగ్ని మూల — వంట అక్కడే చేయాలని వాస్తు చెప్తది. పక్కా వాస్తు కిచెన్.",
                   "आग्नेय कोण अग्नि का कोना है — वास्तु कहता है खाना यहीं बनना चाहिए। एकदम वास्तु वाला किचन।")
    if kind == "KITCHEN":
        return t.p("Kitchen in the North-West — after South-East, Vastu's next good place for cooking.",
                   "వాయువ్యంలో కిచెన్ — ఆగ్నేయం తర్వాత వంటకి వాస్తులో రెండో మంచి ప్లేస్ ఇదే.",
                   "वायव्य कोण में किचन — आग्नेय के बाद वास्तु में खाना बनाने की दूसरी अच्छी जगह यही है।")
    if kind == "MASTER_BEDROOM":
        return t.p("South-West is for the head of the family — Vastu says sleeping here keeps the home steady.",
                   "నైరుతి ఇంటి పెద్ద దిక్కు — ఇక్కడ పడుకుంటే ఇంట్లో స్థిరత్వం ఉంటది అని వాస్తు చెప్తది.",
                   "नैऋत्य कोण घर के मुखिया का है — वास्तु कहता है यहाँ सोने से घर में स्थिरता रहती है।")
    return t.p(f"Hall {loc} — as per Vastu very good, the house feels light and open.",
               f"హాల్ {loc} — వాస్తు ప్రకారం చాలా మంచిది, ఇల్లంతా వెలుతురు, పాజిటివ్ గా ఉంటది.",
               f"हॉल {loc} — वास्तु के हिसाब से बहुत बढ़िया, घर खुला-खुला और रोशन रहता है।")


def _say_highlight(t: T, f: Dict, c: Dict, pid: str, a: Dict) -> Optional[str]:
    """One highlight, in words, looking at what it is about."""
    kind = f["kind"]
    loc = LOC[t.lang].get(f.get("zone") or "", "")
    if pid == "largest":
        return C("room", _pick(t, f, pid, a, [
            ("The biggest bedroom in the house — nice and spacious.", "ఇంట్లో అన్నిటికంటే పెద్ద బెడ్రూమ్ ఇదే — బాగా విశాలంగా ఉంటది.",
             "घर का सबसे बड़ा बेडरूम यही है — खुला-खुला, बड़ा।"),
            ("Of all the bedrooms, this one is the biggest — plenty of room.", "అన్ని బెడ్రూమ్స్ లో ఇదే పెద్దది — కావాల్సినంత స్థలం.",
             "सारे बेडरूम में ये सबसे बड़ा है — भरपूर जगह।")]))
    if pid == "roomy":
        if kind == "LIVING":
            return C("room", t.p("Nice and big — festivals, functions, relatives, everyone fits.",
                                 "బాగా పెద్దది — పండగలు, ఫంక్షన్లు, చుట్టాలు ఎంతమంది వచ్చినా సరిపోతది.",
                                 "खूब बड़ा — त्योहार हो, फ़ंक्शन हो, मेहमान कितने भी आएँ, जगह कम नहीं।"))
        if kind == "KITCHEN":
            return C("counter", t.p("A big kitchen — two people can cook together comfortably.",
                                    "పెద్ద కిచెన్ — ఇద్దరు కలిసి వంట చేసినా ఇరుకు ఉండదు.",
                                    "बड़ा किचन — दो लोग साथ में आराम से खाना बना सकते हैं।"))
        if kind.endswith("BEDROOM"):
            return C("room", _pick(t, f, pid, a, [
                ("Spacious — you can walk around the bed freely.", "విశాలంగా ఉంటది — బెడ్ చుట్టూ హాయిగా తిరగొచ్చు.",
                 "खुला-खुला कमरा — बेड के चारों तरफ़ आराम से घूम सकते हैं।"),
                ("A good big room — it never feels cramped.", "మంచి పెద్ద గది — ఎప్పుడూ ఇరుకు అనిపించదు.",
                 "अच्छा बड़ा कमरा — कभी तंग नहीं लगता।")]))
        return C("room", t.p("Big and open — the whole family together.", "బాగా విశాలంగా — ఫ్యామిలీ అంతా కలిసి కూర్చోవచ్చు.",
                             "बड़ा और खुला — पूरा परिवार एक साथ।"))
    if pid == "bath_roomy":
        return C("room", _pick(t, f, pid, a, [
            ("Roomy for a bathroom — easy to move about.", "బాత్రూమ్ అయినా విశాలంగా ఉంటది — ఇరుకు అనిపించదు.",
             "बाथरूम होकर भी खुला-खुला — तंगी नहीं लगती।"),
            ("Plenty of space, even in the bathroom.", "బాత్రూమ్ లో కూడా కావాల్సినంత స్థలం.", "बाथरूम में भी भरपूर जगह।")]))
    if pid == "v_special":
        return C("room", _special_vastu(t, kind, f))
    if pid == "v_best":
        return C("room", t.p(f"It is {loc} — as per Vastu, the best place for it.",
                             f"ఇది {loc} ఉంది — వాస్తు ప్రకారం దీనికి ఇదే బెస్ట్ ప్లేస్.",
                             f"ये {loc} है — वास्तु के हिसाब से इसके लिए यही सबसे अच्छी जगह है।"))
    if pid == "v_good":
        return C("room", t.p(f"It is {loc} — a good place in Vastu also.", f"ఇది {loc} ఉంది — వాస్తుకి కూడా మంచి ప్లేస్.",
                             f"ये {loc} है — वास्तु के हिसाब से भी अच्छी जगह।"))
    if pid == "v_bath":
        return C("room", _pick(t, f, pid, a, [
            ("Vastu-wise too, this is the right direction for a bathroom.", "వాస్తు ప్రకారం కూడా బాత్రూమ్ కి ఇదే సరైన దిక్కు.",
             "वास्तु के हिसाब से भी बाथरूम के लिए यही सही दिशा है।"),
            ("And it sits where Vastu wants a bathroom to be.", "వాస్తు చెప్పిన చోటే బాత్రూమ్ ఉంది.",
             "और बाथरूम वहीं है जहाँ वास्तु कहता है।")]))
    if pid == "cross":
        sides = t.join([DIR[t.lang].get(s, s) for s in a["sides"]])
        return C("win:" + a["sides"][0], _pick(t, f, pid, a, [
            (f"Windows on two sides — {sides}. Cross ventilation, the breeze flows right through.",
             f"రెండు వైపులా కిటికీలు — {sides}. గాలి అటు ఇటు ధారాళంగా ఆడుతది.",
             f"दो तरफ़ खिड़कियाँ — {sides}। हवा आर-पार खुलकर चलती है।"),
            (f"Light and air from two sides, {sides} — the room never feels closed.",
             f"{sides} — రెండు వైపుల నుంచి వెలుతురు, గాలి; గది ఎప్పుడూ ఉక్కపోతగా ఉండదు.",
             f"{sides} — दोनों तरफ़ से रोशनी और हवा; कमरा कभी बंद-बंद नहीं लगता।")]))
    if pid == "east":
        return C("win:E", _pick(t, f, pid, a, [
            ("Window on the East — the morning sun comes straight in.", "తూర్పు కిటికీ — పొద్దున్నే లేలేత ఎండ నేరుగా లోపలికి వస్తది.",
             "पूर्व की खिड़की — सुबह की धूप सीधे अंदर आती है।"),
            ("The first light of the morning comes into this room — every day starts fresh.",
             "పొద్దున్న మొదటి వెలుగు ఈ గదిలోకే వస్తది — రోజు ఫ్రెష్ గా మొదలవుతది.",
             "सुबह की पहली रोशनी इसी कमरे में आती है — हर दिन ताज़गी से शुरू होता है।")]))
    if pid == "north":
        return C("win:N", _pick(t, f, pid, a, [
            ("Window on the North — soft, even light all day; perfect for reading.",
             "ఉత్తరం కిటికీ — రోజంతా చల్లని, సమానమైన వెలుతురు; చదువుకోవడానికి సూపర్.",
             "उत्तर की खिड़की — दिन भर नरम, एक जैसी रोशनी; पढ़ाई के लिए बढ़िया।"),
            ("North light — bright, but never harsh, and the room stays cool.",
             "ఉత్తరం వెలుతురు — బాగా వెలుగుంటది కానీ ఎండ తగలదు, గది చల్లగా ఉంటది.",
             "उत्तर की रोशनी — उजाला भरपूर, पर धूप की तपिश नहीं; कमरा ठंडा रहता है।")]))
    if pid == "south":
        return C("win:S", _pick(t, f, pid, a, [
            ("Window on the South — bright all day, and warm sun in the winter.",
             "దక్షిణం కిటికీ — పగలంతా వెలుతురు, చలికాలంలో వెచ్చని ఎండ.", "दक्षिण की खिड़की — दिन भर उजाला, सर्दियों में गुनगुनी धूप।"),
            ("Daylight from the South all day long — never a dull corner.",
             "దక్షిణం నుంచి రోజంతా వెలుతురు — ఏ మూలా చీకటిగా ఉండదు.", "दक्षिण से दिन भर रोशनी — कोई कोना अँधेरा नहीं।")]))
    if pid == "west":
        return C("win:W", _pick(t, f, pid, a, [
            ("Window on the West — golden evening light, the sunset right here.",
             "పడమర కిటికీ — సాయంత్రం బంగారు రంగు వెలుతురు, సూర్యాస్తమయం ఇక్కడ నుంచే.",
             "पश्चिम की खिड़की — शाम की सुनहरी रोशनी, सूर्यास्त यहीं से।"),
            ("Evenings, this room fills with soft golden light from the West.",
             "సాయంత్రం అయితే పడమర నుంచి మెత్తటి బంగారు వెలుగు గది నిండా.",
             "शाम को पश्चिम से नरम सुनहरी रोशनी पूरे कमरे में भर जाती है।")]))
    if pid == "inner":
        return C("room", _pick(t, f, pid, a, [
            ("Tucked inside the house — cool, quiet and private.", "ఇంటి లోపల వైపు — చల్లగా, ప్రశాంతంగా, ప్రైవేట్ గా ఉంటది.",
             "घर के अंदर की तरफ़ — ठंडा, शांत और प्राइवेट।"),
            ("Away from the outside walls — no sun heat, no street noise.", "బయటి గోడలకి దూరంగా — ఎండ వేడి లేదు, రోడ్డు శబ్దం లేదు.",
             "बाहर की दीवारों से दूर — न धूप की गर्मी, न सड़क का शोर।")]))
    if pid == "bath_window":
        return C("win:any", _pick(t, f, pid, a, [
            ("A window of its own — fresh air, and it dries quickly.",
             "సొంత కిటికీ ఉంది — గాలి ఆడుతది, త్వరగా ఆరిపోతది, ఎప్పుడూ ఫ్రెష్ గా ఉంటది.",
             "अपनी खिड़की है — ताज़ी हवा, और जल्दी सूख जाता है।"),
            ("Daylight and fresh air come straight in — it stays dry and fresh.",
             "వెలుతురు, గాలి నేరుగా వస్తాయి — ఎప్పుడూ పొడిగా, ఫ్రెష్ గా.", "रोशनी और ताज़ी हवा सीधे अंदर — हमेशा सूखा और ताज़ा।")]))
    if pid == "vent":
        return C("win:any", _pick(t, f, pid, a, [
            ("A ventilator up high — no smell ever stays inside.", "పైన వెంటిలేటర్ — గాలి బయటికి పోతది, వాసన అనే మాటే ఉండదు.",
             "ऊपर वेंटिलेटर — हवा बाहर निकलती है, बदबू का नामो-निशान नहीं।"),
            ("Air goes out through the ventilator up there — always fresh.", "అక్కడ పైన వెంటిలేటర్ నుంచి గాలి పోతది — ఎప్పుడూ ఫ్రెష్.",
             "ऊपर के वेंटिलेटर से हवा निकलती है — हमेशा ताज़ा।")]))
    if pid == "fan":
        return C("room", _pick(t, f, pid, a, [
            ("An exhaust fan keeps it fresh and dry.", "ఎగ్జాస్ట్ ఫ్యాన్ ఉంటది — ఎప్పుడూ ఫ్రెష్ గా, డ్రైగా.",
             "एग्ज़ॉस्ट फ़ैन से हमेशा ताज़ा और सूखा रहता है।"),
            ("With the exhaust fan on, it dries in minutes.", "ఎగ్జాస్ట్ ఫ్యాన్ వేస్తే నిమిషాల్లో పొడిగా అయిపోతది.",
             "एग्ज़ॉस्ट फ़ैन चलाओ तो मिनटों में सूख जाता है।")]))
    if pid == "bed":
        b = _tr(t, _BED[a["bed"]])
        wr = a.get("wardrobe")
        return C("bed", _pick(t, f, pid, a, [
            (f"{cap(b)} bed fits easily" + (" — and a wardrobe too." if wr else "."),
             f"{b} బెడ్ సులువుగా పడతది" + (" — వార్డ్రోబ్ కూడా." if wr else "."),
             f"{b} बेड आराम से आ जाएगा" + (" — वॉर्डरोब भी।" if wr else "।")),
            (f"A {b} bed, with space to walk all around it" + (" — and the wardrobe on the wall." if wr else "."),
             f"{b} బెడ్ వేసినా చుట్టూ నడవడానికి బోలెడంత స్థలం" + (" — గోడకి వార్డ్రోబ్ కూడా." if wr else "."),
             f"{b} बेड लगाकर भी चारों तरफ़ चलने की जगह" + (" — दीवार पर वॉर्डरोब भी।" if wr else "।"))]))
    if pid == "desk":
        return C("desk", t.p("A study table fits too — a quiet corner to read.", "స్టడీ టేబుల్ కూడా పడతది — ప్రశాంతంగా చదువుకోవడానికి.",
                             "स्टडी टेबल भी आ जाएगी — शांति से पढ़ने के लिए।"))
    if pid == "sofa":
        n = a["n"]
        return C("sofa", t.p(f"{t.i(n)} seater sofa and TV unit — even when relatives come, no problem.",
                             f"{t.i(n)} సీటర్ సోఫా, టీవీ యూనిట్ — చుట్టాలు వచ్చినా ఇరుకుగా ఉండదు.",
                             f"{t.i(n)} सीटर सोफ़ा और टीवी यूनिट — मेहमान आएँ तो भी जगह की कमी नहीं।"))
    if pid == "dining":
        n = a["n"]
        return C("dining", t.p(f"Dining table for {t.i(n)} — the whole family eats together.",
                               f"{t.i(n)} మంది కూర్చునే డైనింగ్ టేబుల్ — ఫ్యామిలీ అంతా కలిసి తినొచ్చు.",
                               f"{t.i(n)} लोगों की डाइनिंग टेबल — पूरा परिवार साथ बैठकर खाए।"))
    if pid == "counter":
        m = a["m"]
        return C("counter", t.p(f"{t.h(m)} metres of counter — stove, sink and fridge, all have their place.",
                                f"{t.h(m)} మీటర్ల కౌంటర్ — స్టవ్, సింక్, ఫ్రిడ్జ్ అన్నిటికీ ప్లేస్ ఉంది.",
                                f"{t.h(m)} मीटर का काउंटर — चूल्हा, सिंक, फ़्रिज सबकी जगह है।"))
    if pid == "washer":
        return C("washer", t.p("A place for the washing machine also.", "వాషింగ్ మెషిన్ కి కూడా సెపరేట్ ప్లేస్ ఉంది.",
                               "वॉशिंग मशीन की जगह भी है।"))
    if pid == "fixtures":
        parts = [w for k, w in (("wc", t.p("toilet", "టాయిలెట్", "टॉयलेट")), ("shower", t.p("shower", "షవర్", "शावर")),
                                ("basin", t.p("wash basin", "వాష్ బేసిన్", "वॉश बेसिन"))) if a.get(k)]
        return C("wc", _pick(t, f, pid, a, [
            (f"{cap(t.join(parts))} — everything inside.", f"లోపల {t.join(parts)} — అన్నీ ఇందులోనే.",
             f"अंदर {t.join(parts)} — सब कुछ यहीं।"),
            (f"{cap(t.join(parts))} — all fitted in neatly.", f"{t.join(parts)} — అన్నీ పొందిగ్గా అమరాయి.",
             f"{t.join(parts)} — सब सलीके से फ़िट।")]))
    if pid == "serve":
        return C("door", t.p("From the kitchen straight to the dining — serve the food hot.",
                             "కిచెన్ నుంచి డైనింగ్ కి నేరుగా — వేడి వేడిగా వడ్డించొచ్చు.",
                             "किचन से सीधे डाइनिंग — खाना गरमा-गरम परोसिए।"))
    if pid == "utility":
        return C("door", t.p("The washing area is right behind — vessels and clothes never come into the kitchen.",
                             "వెనకాలే యుటిలిటీ — అంట్లు, బట్టలు కిచెన్ లోకి రావు, కిచెన్ ఎప్పుడూ నీట్ గా ఉంటది.",
                             "पीछे ही यूटिलिटी — बर्तन और कपड़े किचन में नहीं आते, किचन हमेशा साफ़।"))
    if pid == "utility_back":
        return C("room", t.p("Right behind the kitchen — the washing never comes into the cooking space.",
                             "కిచెన్ వెనకాలే — ఉతకడం, తోమడం అంతా ఇక్కడే; కిచెన్ శుభ్రంగా ఉంటది.",
                             "किचन के ठीक पीछे — धुलाई सब यहीं, किचन साफ़-सुथरा रहता है।"))
    if pid == "dining_kitchen":
        return C("door", t.p("The kitchen opens right here — hot food straight to the table.",
                             "కిచెన్ పక్కనే — వంట అయిన వెంటనే వేడిగా టేబుల్ మీదకి.",
                             "किचन बिल्कुल बगल में — खाना सीधे गरमा-गरम टेबल पर।"))
    if pid == "privacy":
        return C("door", t.p("With the foyer in front, people at the door never see straight into the hall.",
                             "ముందు ఫోయర్ ఉంది — తలుపు తీసినా బయటి వాళ్ళకి హాల్ కనిపించదు.",
                             "आगे फ़ॉयर है — दरवाज़ा खुले तो भी बाहर वालों को हॉल नहीं दिखता।"))
    if pid == "pooja_hall":
        return C("door", t.p("Right off the hall — the whole family prays together.",
                             "హాల్ పక్కనే — ఇంట్లో అందరూ కలిసి పూజ చేసుకోవచ్చు.",
                             "हॉल के बगल में — पूरा परिवार साथ में पूजा कर सके।"))
    if pid == "quiet":
        return C("room", _pick(t, f, pid, a, [
            ("At the back, away from the road — quiet, peaceful sleep.",
             "రోడ్డుకి దూరంగా, ఇంటి వెనక వైపు — శబ్దం ఉండదు, హాయిగా నిద్ర పడతది.",
             "सड़क से दूर, घर के पीछे — शोर नहीं, सुकून की नींद।"),
            ("Far from the road noise — you sleep in peace here.", "రోడ్డు శబ్దం ఇక్కడికి రాదు — ప్రశాంతంగా నిద్రపోవచ్చు.",
             "सड़क का शोर यहाँ तक नहीं आता — चैन की नींद।")]))
    if pid == "near_parents":
        return C("room", t.p("The master bedroom is right next door — the children stay close to you.",
                             "పక్కనే మాస్టర్ బెడ్రూమ్ — పిల్లలు మీకు దగ్గరగానే ఉంటారు.",
                             "बगल में ही मास्टर बेडरूम — बच्चे आपके पास ही रहेंगे।"))
    if pid == "elder_ground":
        return C("room", t.p("It is on the ground floor — no stairs to climb for them.",
                             "గ్రౌండ్ ఫ్లోర్ లోనే — వాళ్ళు మెట్లు ఎక్కాల్సిన పనే లేదు.",
                             "ग्राउंड फ़्लोर पर ही है — उन्हें सीढ़ियाँ नहीं चढ़नी पड़ेंगी।"))
    if pid == "elder_lift":
        return C("room", t.p("The lift is there, so the stairs are no trouble for them.",
                             "లిఫ్ట్ ఉంది కాబట్టి వాళ్ళకి మెట్ల ఇబ్బంది లేదు.",
                             "लिफ़्ट है, तो उन्हें सीढ़ियों की परेशानी नहीं।"))
    if pid == "night":
        return C("door", _pick(t, f, pid, a, [
            ("At night, no need to step out of the bedroom.", "రాత్రి పూట గది దాటి బయటికి రావాల్సిన పనే లేదు.",
             "रात को कमरे से बाहर निकलने की ज़रूरत ही नहीं।"),
            ("Only for this room — nobody else uses it.", "ఈ గది వాళ్ళకి మాత్రమే — వేరే ఎవరూ వాడరు.",
             "सिर्फ़ इसी कमरे वालों का — कोई और इस्तेमाल नहीं करता।")]))
    if pid == "elder_safe":
        return C("door", t.p("Right beside the elders' bed — safe and easy for them, even at night.",
                             "పెద్దవాళ్ళ బెడ్ పక్కనే — రాత్రి పూట కూడా వాళ్ళకి సేఫ్ గా, ఈజీగా.",
                             "बड़ों के बिस्तर के पास ही — रात में भी उनके लिए सुरक्षित और आसान।"))
    if pid == "guests":
        return C("door", t.p("Near the hall — handy for guests, and the bedrooms stay private.",
                             "హాల్ కి దగ్గరగా — చుట్టాలకి ఈజీ, బెడ్రూమ్స్ ప్రైవసీ కూడా ఉంటది.",
                             "हॉल के पास — मेहमानों के लिए आसान, बेडरूम की प्राइवेसी भी बनी रहे।"))
    if pid == "pipes":
        return C("wc", t.p("Its pipes share one wall with the room next door — fewer leaks, easy repairs.",
                           "పక్క గదితో కలిపి ఒకే గోడలో పైపులు — లీకేజీలు తక్కువ, రిపేర్ కూడా ఈజీ.",
                           "पाइप बगल के कमरे के साथ एक ही दीवार में — लीकेज कम, मरम्मत आसान।"))
    if pid == "dress":
        return C("room", _pick(t, f, pid, a, [
            ("Wardrobes all along — all your clothes in one place, and the bedroom stays neat.",
             "బట్టలన్నీ ఇక్కడే — బెడ్రూమ్ ఎప్పుడూ నీట్ గా ఉంటది.", "सारे कपड़े एक जगह — और बेडरूम हमेशा साफ़-सुथरा।"),
            ("A room just for clothes — get ready in peace, and the bedroom stays tidy.",
             "బట్టల కోసమే ఒక గది — హాయిగా రెడీ అవ్వొచ్చు, బెడ్రూమ్ చక్కగా ఉంటది.",
             "कपड़ों के लिए अलग कमरा — आराम से तैयार होइए, बेडरूम साफ़-सुथरा।")]))
    return None


def _room(t: T, f: Dict, c: Dict) -> List[str]:
    """A room in its best light: welcome, its strongest highlights, and — a bedroom with its own bath — on into it."""
    kind = f["kind"]
    out = []
    if f.get("first_up"):
        out.append(t.p("Now we are upstairs.", "ఇప్పుడు పైకి వచ్చాం.", "अब हम ऊपर आ गए।"))
    out.append(C("room", _intro(t, f, c)))
    hl = f.get("_hl")
    if hl is None:
        hl = room_highlights(dict(f, lift=bool(c.get("lift"))))
    for pid, a in hl:
        s = _say_highlight(t, f, c, pid, a)
        if s:
            out.append(s)
    if kind.endswith("BEDROOM") and f.get("attached"):
        if f.get("bath_next"):
            dr = f.get("has_dress")
            out.append(C("bath", _pick(t, f, "tease", {"_n": f.get("tease_n")}, [
                ("And it has its own bathroom" + (" and dressing room" if dr else "") + " — come, let me show you.",
                 "దీనికి సొంత బాత్రూమ్" + (", డ్రెస్సింగ్ రూమ్" if dr else "") + " ఉంది అండీ — రండి, అది కూడా చూపిస్తా.",
                 "इसका अपना बाथरूम" + (" और ड्रेसिंग रूम" if dr else "") + " भी है — आइए, वो भी दिखाता हूँ।"),
                ("Its own " + ("dressing room and bathroom are" if dr else "bathroom is") + " right inside — let me show you.",
                 "లోపలే సొంత " + ("డ్రెస్సింగ్ రూమ్, బాత్రూమ్" if dr else "బాత్రూమ్") + " ఉన్నాయి — చూపిస్తా రండి."
                 if dr else "లోపలే సొంత బాత్రూమ్ ఉంది — చూపిస్తా రండి.",
                 "अंदर ही अपना " + ("ड्रेसिंग रूम और बाथरूम" if dr else "बाथरूम") + " है — चलिए, दिखाता हूँ।")])))
        else:
            out.append(C("bath", t.p("Bathroom is attached — no need to go outside at night.",
                                     "బాత్రూమ్ అటాచ్డ్ — రాత్రి పూట బయటికి వెళ్ళాల్సిన పని లేదు.",
                                     "बाथरूम अटैच्ड है — रात को बाहर जाने की ज़रूरत नहीं।")))
    return out


_SPECIAL = {
    "kitchen_se": ("the kitchen in the Agni corner", "అగ్ని మూలలో కిచెన్", "आग्नेय कोण में किचन"),
    "master_sw": ("the master bedroom in the South-West", "నైరుతిలో మాస్టర్ బెడ్రూమ్", "नैऋत्य कोण में मास्टर बेडरूम"),
    "pooja_ne": ("the pooja room in the Ishan corner", "ఈశాన్యంలో పూజ గది", "ईशान कोण में पूजा घर"),
    "door": ("the main door in a lucky pada", "శుభ పాదంలో మెయిన్ డోర్", "शुभ पद में मेन डोर"),
}


def _home_highlights(t: T, c: Dict) -> Optional[str]:
    """The closing line: what makes this home special, from what the walk found (three at most)."""
    h = c.get("home_hl") or {}
    items = [_tr(t, _SPECIAL[k]) for k in ("door", "kitchen_se", "master_sw", "pooja_ne") if h.get(k)]
    if h.get("cross", 0) >= 2:
        items.append(t.p(f"cross ventilation in {t.i(h['cross'])} rooms", f"{t.i(h['cross'])} గదుల్లో క్రాస్ వెంటిలేషన్",
                         f"{t.i(h['cross'])} कमरों में क्रॉस वेंटिलेशन"))
    if h.get("baths", 0) >= 2:
        items.append(t.p(f"{t.i(h['baths'])} attached bathrooms", f"{t.i(h['baths'])} అటాచ్డ్ బాత్రూమ్స్",
                         f"{t.i(h['baths'])} अटैच्ड बाथरूम"))
    if h.get("east", 0) >= 2:
        items.append(t.p(f"morning sun in {t.i(h['east'])} rooms", f"{t.i(h['east'])} గదుల్లో పొద్దున్న ఎండ",
                         f"{t.i(h['east'])} कमरों में सुबह की धूप"))
    items = items[:3]
    if not items:
        return None
    return t.p(f"What makes this home special: {t.join(items)}.", f"ఈ ఇంటి స్పెషల్ ఏంటంటే — {t.join(items)}.",
               f"इस घर की ख़ासियत — {t.join(items)}।")


def _plot_line(t: T, c: Dict) -> str:
    area = c["plot_m2"]
    return t.p(f"This is a {t.h(area / M2_PER_CENT)} cent plot — about {t.i(area * SQFT)} square feet.",
               f"ఇది {t.h(area / M2_PER_CENT)} సెంట్ల స్థలం — అంటే దాదాపు {t.i(area * GAJ)} గజాలు.",
               f"ये {t.i(area * GAJ)} गज का प्लॉट है — यानी लगभग {t.i(area * SQFT)} स्क्वेयर फ़ीट।")


def _facing_lines(t: T, c: Dict) -> List[str]:
    fc = c.get("facing", "N")
    face = FACING[t.lang].get(fc, fc)
    if c.get("kind") in ("apartment", "floors"):
        out = [t.p(f"The building is {face} facing.", f"బిల్డింగ్ {face} ఫేసింగ్.", f"बिल्डिंग {face} फ़ेसिंग है।")]
    else:
        out = [t.p(f"The house is {face} facing.", f"ఇల్లు {face} ఫేసింగ్.", f"घर {face} फ़ेसिंग है।")]
    if fc in ("E", "NE"):
        out.append(t.p("East facing — the one most families ask for. Morning sun comes right to the front.",
                       "ఈస్ట్ ఫేసింగ్ అంటే అందరూ కోరుకునేది — పొద్దున్నే సూర్యుడు ఇంటి ముందే.",
                       "ईस्ट फ़ेसिंग सबकी पहली पसंद होता है — सुबह का सूरज सीधे घर के सामने।"))
    elif fc in ("N", "NW"):
        out.append(t.p("North is Kubera's direction in Vastu — the direction of wealth.",
                       "ఉత్తరం కుబేరుడి దిక్కు అని వాస్తు చెప్తది — సంపదకి దిక్కు.",
                       "उत्तर कुबेर की दिशा मानी जाती है — धन की दिशा।"))
    elif fc in ("S", "SE", "SW"):
        out.append(t.p("South facing — bright sunlight at the front of the house all day long.",
                       "సౌత్ ఫేసింగ్ — ఇంటి ముందు రోజంతా మంచి వెలుతురు.",
                       "साउथ फ़ेसिंग — घर के सामने दिन भर अच्छी रोशनी।"))
    else:
        out.append(t.p("West facing — golden evening light at the front, and the rooms at the back stay cool in the afternoon.",
                       "వెస్ట్ ఫేసింగ్ — సాయంత్రం ఇంటి ముందు బంగారు వెలుతురు, మధ్యాహ్నం వెనక గదులు చల్లగా ఉంటాయి.",
                       "वेस्ट फ़ेसिंग — शाम को घर के सामने सुनहरी रोशनी, दोपहर में पीछे के कमरे ठंडे रहते हैं।"))
    if fc not in ("E", "NE", "N", "NW") and c.get("door_lucky"):
        out.append(t.p("And the main door stands in a lucky pada — I will show you.",
                       "పైగా మెయిన్ డోర్ శుభ పాదంలో ఉంది — చూపిస్తా రండి.",
                       "और मेन डोर शुभ पद में है — दिखाता हूँ।"))
    return out


def _gate_lines(t: T, g: Dict) -> List[str]:
    out = []
    car, walk, wicket, fire, car_cls = g.get("car_ft"), g.get("walk_ft"), g.get("wicket"), g.get("fire"), g.get("car")
    op = g.get("op")
    if car:
        big = car_cls in ("suv", "muv")
        what = t.p("even a big SUV" if big else "your car", "పెద్ద ఎస్యూవీ కారు కూడా" if big else "మీ కారు",
                   "बड़ी एसयूवी भी" if big else "आपकी गाड़ी")
        out.append(t.p(f"See the gate — {t.h(car)} feet. {cap(what)} comes in straight, without a scratch.",
                       f"గేట్ చూడండి — {t.h(car)} అడుగులు. {what} ఎక్కడా తగలకుండా నేరుగా లోపలికి వస్తది.",
                       f"गेट देखिए — {t.h(car)} फ़ीट। {what} बिना खरोंच सीधे अंदर आ जाएगी।"))
        if fire:
            out.append(t.p("Wide enough for a fire engine — for safety.",
                           "ఫైర్ ఇంజన్ కూడా లోపలికి వచ్చేంత వెడల్పు — సేఫ్టీ కోసం.",
                           "फ़ायर ब्रिगेड की गाड़ी भी अंदर आ सके, इतना चौड़ा — सुरक्षा के लिए।"))
        if op == "sliding":
            out.append(t.p("It is a sliding gate — it rolls to the side, no space wasted.",
                           "ఇది స్లైడింగ్ గేట్ — పక్కకి జరుగుతది, స్థలం వృథా కాదు.",
                           "ये स्लाइडिंग गेट है — साइड में खिसकता है, जगह बर्बाद नहीं होती।"))
        elif op == "telescopic":
            out.append(t.p("Telescopic gate — it slides in two parts, half the space is enough.",
                           "టెలిస్కోపిక్ గేట్ — రెండు ముక్కలుగా పక్కకి జరుగుతది, సగం స్థలం చాలు.",
                           "टेलिस्कोपिक गेट — दो हिस्सों में खिसकता है, आधी जगह काफ़ी।"))
        elif op == "bifold":
            out.append(t.p("Folding gate — it folds and opens in very little space.",
                           "మడిచే గేట్ — మడతపడి తెరుచుకుంటది, తక్కువ స్థలం చాలు.",
                           "फ़ोल्डिंग गेट — मुड़कर खुलता है, कम जगह में।"))
        else:
            out.append(t.p("The gate opens inside — never onto the road.",
                           "గేట్ లోపలికే తెరుచుకుంటది — రోడ్డు మీదకి రాదు.",
                           "गेट अंदर की तरफ़ खुलता है — सड़क पर नहीं।"))
        if wicket:
            out.append(t.p("Inside the big gate there is a small door for walking — no need to open the big gate every time.",
                           "పెద్ద గేట్ లోనే నడిచి వెళ్ళడానికి చిన్న డోర్ ఉంది — ప్రతిసారీ పెద్ద గేట్ తెరవక్కర్లేదు.",
                           "बड़े गेट में ही पैदल आने-जाने का छोटा दरवाज़ा है — हर बार बड़ा गेट खोलने की ज़रूरत नहीं।"))
        elif walk:
            out.append(t.p(f"Next to it, a {t.h(walk)} feet small gate for walking.",
                           f"పక్కనే {t.h(walk)} అడుగుల చిన్న గేట్ — నడిచి వెళ్ళడానికి.",
                           f"बगल में {t.h(walk)} फ़ीट का छोटा गेट — पैदल आने-जाने के लिए।"))
    elif g.get("bike_ft"):
        out.append(t.p(f"The gate is {t.h(g['bike_ft'])} feet — bikes and scooters come in easily.",
                       f"గేట్ {t.h(g['bike_ft'])} అడుగులు — బైక్, స్కూటర్ సులువుగా లోపలికి వస్తాయి.",
                       f"गेट {t.h(g['bike_ft'])} फ़ीट का है — बाइक, स्कूटी आराम से अंदर आ जाएँगी।"))
    elif walk:
        out.append(t.p(f"A {t.h(walk)} feet gate for walking in.", f"{t.h(walk)} అడుగుల గేట్ — నడిచి రావడానికి.",
                       f"{t.h(walk)} फ़ीट का गेट — पैदल आने के लिए।"))
    return out


def _family_line(t: T, c: Dict) -> Optional[str]:
    m = int(c.get("members") or 0)
    if not m:
        return None
    return t.p(f"Your family is {t.i(m)} people — this home is planned for you.",
               f"మీ ఇంట్లో {t.i(m)} మంది — ఈ ఇల్లు మీ ఫ్యామిలీ కోసమే ప్లాన్ చేశాం.",
               f"आपके परिवार में {t.i(m)} लोग हैं — ये घर आपके हिसाब से ही बना है।")


def _start(t: T, f: Dict, c: Dict) -> List[str]:
    out = [t.p("Namaste! Welcome, welcome.", "నమస్కారం అండీ! రండి, రండి.", "नमस्ते जी! आइए, आइए।")]
    kind = c["kind"]
    bhk = t.bhk(c.get("bhk", ""))
    if kind in ("apartment", "floors"):
        floors, flats = int(c.get("floors") or 1), int(c.get("flats") or 1)
        out.append(_plot_line(t, c))
        if kind == "apartment":
            homes = floors * flats
            out.append(t.p(f"On it stands this apartment — {t.i(floors)} floors, {t.i(flats)} flat{'s' if flats > 1 else ''} "
                           f"on each floor, {t.i(homes)} families in all.",
                           f"ఇందులో కట్టిన అపార్ట్మెంట్ — {t.i(floors)} ఫ్లోర్లు, ఒక్కో ఫ్లోర్ కి {t.i(flats)} "
                           f"{'ఫ్లాట్' if flats == 1 else 'ఫ్లాట్లు'}, మొత్తం {t.i(homes)} ఫ్యామిలీలు.",
                           f"इस पर ये अपार्टमेंट है — {t.i(floors)} फ़्लोर, हर फ़्लोर पर {t.i(flats)} फ़्लैट, "
                           f"कुल {t.i(homes)} परिवार।"))
            out.append(t.p(f"Every flat is a {bhk} of {t.i(c.get('flat_sqft', 0))} square feet carpet.",
                           f"ప్రతి ఫ్లాట్ {bhk} — {t.i(c.get('flat_sqft', 0))} స్క్వేర్ ఫీట్ కార్పెట్.",
                           f"हर फ़्लैट {bhk} है — {t.i(c.get('flat_sqft', 0))} स्क्वेयर फ़ीट कारपेट।"))
        else:
            out.append(t.p(f"This building has {t.i(floors)} floors — a {bhk} home on every floor, each with its own main door.",
                           f"ఇది {t.i(floors)} ఫ్లోర్ల బిల్డింగ్ — ప్రతి ఫ్లోర్ లో ఒక {bhk} ఇల్లు, దేనికది సెపరేట్ మెయిన్ డోర్.",
                           f"ये {t.i(floors)} फ़्लोर की बिल्डिंग है — हर फ़्लोर पर एक {bhk} घर, सबका अलग मेन डोर।"))
            out.append(t.p("You live on one floor and give the others on rent — every month, rent comes in.",
                           "మీరు ఒక ఫ్లోర్ లో ఉండి, మిగతావి అద్దెకి ఇవ్వొచ్చు — ప్రతి నెలా అద్దె వస్తది.",
                           "आप एक फ़्लोर पर रहिए, बाक़ी किराए पर दीजिए — हर महीने किराया आएगा।"))
        out += _facing_lines(t, c)
        out.append(t.p("Come, let us start from the gate.", "రండి, గేట్ నుంచి మొదలుపెడదాం.", "चलिए, गेट से शुरू करते हैं।"))
        return out
    out.append(_plot_line(t, c))
    out += _facing_lines(t, c)
    if c.get("road_ft"):
        out.append(C("road", t.p(f"The road in front is {t.i(c['road_ft'])} feet wide.",
                                 f"ముందు {t.i(c['road_ft'])} అడుగుల రోడ్డు ఉంది.",
                                 f"सामने {t.i(c['road_ft'])} फ़ीट की सड़क है।")))
    beds = int(c.get("beds") or 0)
    if kind == "duplex":
        out.append(t.p(f"It is a duplex — one {bhk} home on two floors. Hall and kitchen below, bedrooms above — full privacy.",
                       f"ఇది డ్యూప్లెక్స్ — ఒకే {bhk} ఇల్లు రెండు ఫ్లోర్లలో. కింద హాల్, కిచెన్; పైన బెడ్రూమ్స్ — ప్రైవసీ బాగుంటది.",
                       f"ये डुप्लेक्स है — एक ही {bhk} घर दो फ़्लोर पर। नीचे हॉल और किचन, ऊपर बेडरूम — पूरी प्राइवेसी।"))
    else:
        out.append(t.p(f"It is a {bhk} house — {t.i(beds)} bedroom{'s' if beds != 1 else ''}, hall and kitchen.",
                       f"ఇది {bhk} ఇల్లు — {t.i(beds)} {'బెడ్రూమ్' if beds == 1 else 'బెడ్రూమ్స్'}, హాల్, కిచెన్.",
                       f"ये {bhk} घर है — {t.i(beds)} बेडरूम, हॉल और किचन।"))
    fam = _family_line(t, c)
    if fam:
        out.append(fam)
    if f.get("gate"):
        out += _gate_lines(t, f["gate"])
    cars = int(c.get("cars") or 0)
    if cars and not c.get("seen_parking"):
        out.append(t.p(f"Car parking is inside the gate — {t.i(cars)} car{'s' if cars > 1 else ''}, safe in your own compound.",
                       f"గేట్ లోపలే కార్ పార్కింగ్ — {t.i(cars)} {'కారు' if cars == 1 else 'కార్లు'}, మీ కాంపౌండ్ లోనే సేఫ్.",
                       f"गेट के अंदर ही कार पार्किंग — {t.i(cars)} {'गाड़ी' if cars == 1 else 'गाड़ियाँ'}, अपने कंपाउंड में सुरक्षित।"))
    if c.get("stilt"):
        out.append(t.p("The car parks on the stilt floor below; the house starts one floor up.",
                       "కారు కింద స్టిల్ట్ ఫ్లోర్ లో పార్క్ చేసుకోవచ్చు; ఇల్లు ఒక ఫ్లోర్ పైన మొదలవుతది.",
                       "गाड़ी नीचे स्टिल्ट फ़्लोर पर खड़ी होगी; घर एक फ़्लोर ऊपर से शुरू होता है।"))
    if c.get("gate_stop"):
        out.append(C("gate", t.p("Come, let us start from the gate.", "రండి, గేట్ నుంచి మొదలుపెడదాం.",
                                 "चलिए, गेट से शुरू करते हैं।")))
    else:
        out.append(C("door", t.p("Come, let us go inside and see.", "రండి, లోపలికి వెళ్ళి చూద్దాం.",
                                 "चलिए, अंदर चलकर देखते हैं।")))
    return out


def _main_door(t: T, f: Dict, c: Dict) -> List[str]:
    pada, score = f.get("pada"), f.get("score")
    out = []
    side = DIR[t.lang].get((pada or "N")[0], "")
    if pada and len(pada) >= 2 and pada[1:].isdigit():
        k = int(pada[1:]) - 1
        o = _PADA_ORD[t.lang][k] if 0 <= k < 8 else ""
        name = _tr(t, PADA_NAME[pada]) if pada in PADA_NAME and (score or 0) >= 0 else ""
        nm = t.p(f" — the {name} pada" if name else "", f" — {name} పాదం" if name else "",
                 f" — {name} पद" if name else "")
        out.append(t.p(f"This is the main door — on the {side} side, in the {o} pada{nm}.",
                       f"ఇదే మెయిన్ డోర్ — {side} వైపు, {o} పాదంలో{nm}.",
                       f"ये रहा मेन डोर — {side} की तरफ़, {o} पद में{nm}।"))
    else:
        out.append(t.p("This is the main door.", "ఇదే మెయిన్ డోర్.", "ये रहा मेन डोर।"))
    if score is not None:
        if score >= 8:
            gift = _tr(t, PADA_GIFT[pada]) if pada in PADA_GIFT else ""
            out.append(t.p("In Vastu, this is a very lucky pada for the main door" + (f" — it is said to bring {gift}." if gift else "."),
                           "వాస్తు ప్రకారం మెయిన్ డోర్ కి ఇది చాలా శుభమైన పాదం" + (f" — {gift} ఇస్తది అని నమ్మకం." if gift else "."),
                           "वास्तु में मेन डोर के लिए ये बहुत शुभ पद है" + (f" — माना जाता है ये {gift} लाता है।" if gift else "।")))
        elif score >= 3:
            out.append(t.p("Vastu-wise, a good pada for the main door.", "వాస్తు పరంగా మెయిన్ డోర్ కి మంచి పాదం.",
                           "वास्तु के हिसाब से मेन डोर के लिए अच्छा पद।"))
    cm = c.get("plinth_cm")
    if cm:
        out.append(t.p(f"Three steps up — {t.i(cm)} centimetres high, so rain water never comes inside.",
                       f"మూడు మెట్లు — {t.i(cm)} సెంటీమీటర్ల ఎత్తు, వర్షం నీళ్ళు లోపలికి రావు.",
                       f"तीन सीढ़ियाँ — {t.i(cm)} सेंटीमीटर ऊँचा, बारिश का पानी अंदर नहीं आता।"))
    return out


def _tag(lines: List[str], first: Optional[str], rest: Optional[str] = None, last: Optional[str] = None,
         last_if: bool = False) -> List[str]:
    """Cue the sentences of a stop: the first one, the others, and (when last_if) the last one."""
    out = []
    for i, x in enumerate(lines):
        if getattr(x, "cue", None):
            out.append(x)
            continue
        cue = first if i == 0 else rest
        if last_if and i == len(lines) - 1:
            cue = last
        out.append(C(cue, x))
    return out


def _lift_words(t: T, c: Dict) -> Optional[str]:
    lf = c.get("lift")
    if not lf:
        return None
    p = int(lf.get("persons") or 4)
    if lf.get("wheelchair"):
        return t.p(f"{t.i(p)} person lift — a wheelchair also fits inside.",
                   f"{t.i(p)} మంది ఎక్కే లిఫ్ట్ — వీల్ చెయిర్ కూడా పడతది.",
                   f"{t.i(p)} लोगों की लिफ़्ट — व्हीलचेयर भी अंदर आ जाती है।")
    return t.p(f"{t.i(p)} person lift.", f"{t.i(p)} మంది ఎక్కే లిఫ్ట్.", f"{t.i(p)} लोगों की लिफ़्ट।")


def _entrance(t: T, f: Dict, c: Dict) -> List[str]:
    lift = bool(c.get("lift"))
    if f.get("stilt"):
        out = [t.p("The stairs" + (" and the lift" if lift else "") + " start right here at the parking — "
                   "you go up without stepping outside, even in the rain.",
                   "మెట్లు" + (", లిఫ్ట్" if lift else "") + " ఇక్కడ పార్కింగ్ నుంచే మొదలవుతాయి — "
                   "వర్షం పడినా బయటికి వెళ్ళకుండా నేరుగా పైకి.",
                   "सीढ़ियाँ" + (" और लिफ़्ट" if lift else "") + " यहीं पार्किंग से शुरू होती हैं — "
                   "बारिश में भी बाहर निकले बिना सीधे ऊपर।")]
    else:
        out = [t.p("This is the building's entrance. The stairs" + (", the lift" if lift else "") +
                   " and every home's door open onto this lobby.",
                   "ఇదే బిల్డింగ్ ఎంట్రన్స్. మెట్లు" + (", లిఫ్ట్" if lift else "") +
                   ", అన్ని ఇళ్ళ డోర్లు ఈ లాబీ లోకే.",
                   "ये बिल्डिंग का एंट्रेंस है। सीढ़ियाँ" + (", लिफ़्ट" if lift else "") +
                   " और सभी घरों के दरवाज़े इसी लॉबी में खुलते हैं।")]
    lw = _lift_words(t, c)
    if lw:
        out.append(lw)
    return out


def _parking(t: T, f: Dict, c: Dict) -> List[str]:
    n, bikes = int(f.get("cars") or 0), int(f.get("bikes") or 0)
    out = []
    if f.get("stilt"):
        if n:
            out.append(t.p(f"The parking is under the building — the stilt floor. {t.i(n)} car{'s' if n > 1 else ''}, "
                           "safe from sun and rain.",
                           f"పార్కింగ్ బిల్డింగ్ కిందే — స్టిల్ట్ ఫ్లోర్. {t.i(n)} {'కారు' if n == 1 else 'కార్లు'}, "
                           "ఎండా వానా తగలదు.",
                           f"पार्किंग बिल्डिंग के नीचे है — स्टिल्ट फ़्लोर। {t.i(n)} {'गाड़ी' if n == 1 else 'गाड़ियाँ'}, "
                           "न धूप, न बारिश।"))
            out.append(t.p("Every car drives straight in, between the columns — no turning, no scratches.",
                           "ప్రతి కారు కాలమ్స్ మధ్య నుంచి నేరుగా లోపలికి — తిప్పాల్సిన పని లేదు, గీతలు పడవు.",
                           "हर गाड़ी कॉलम के बीच से सीधी अंदर — न मोड़ना, न खरोंच।"))
        elif f.get("wanted"):
            out.append(t.p("This is the stilt floor under the building — open and airy; two-wheelers park easily here.",
                           "ఇది బిల్డింగ్ కింద స్టిల్ట్ ఫ్లోర్ — గాలి ధారాళంగా వస్తది; బైకులు హాయిగా పెట్టుకోవచ్చు.",
                           "ये बिल्डिंग के नीचे का स्टिल्ट फ़्लोर है — खुला और हवादार; दोपहिया आराम से खड़े होते हैं।"))
            out.append(t.p("And with the homes one floor up, every flat stays cool and dry.",
                           "ఇళ్ళు ఒక ఫ్లోర్ పైన ఉంటాయి కాబట్టి — ప్రతి ఇల్లు చల్లగా, తేమ లేకుండా ఉంటది.",
                           "घर एक फ़्लोर ऊपर हैं, तो हर घर ठंडा और सूखा रहता है।"))
        else:
            out.append(t.p("This is the stilt floor under the building — open for parking.",
                           "ఇది బిల్డింగ్ కింద స్టిల్ట్ ఫ్లోర్ — పార్కింగ్ కోసం.",
                           "ये बिल्डिंग के नीचे का स्टिल्ट फ़्लोर है — पार्किंग के लिए।"))
    else:
        out.append(t.p(f"Car parking is inside your own compound — safe. {t.i(n)} car{'s fit' if n > 1 else ' fits'}.",
                       f"కార్ పార్కింగ్ మీ కాంపౌండ్ లోపలే — సేఫ్. {t.i(n)} {'కారు' if n == 1 else 'కార్లు'} పడతాయి.",
                       f"कार पार्किंग आपके कंपाउंड के अंदर ही — सुरक्षित। {t.i(n)} {'गाड़ी' if n == 1 else 'गाड़ियाँ'} आ जाती हैं।"))
    if bikes:
        out.append(t.p(f"Place for {t.i(bikes)} two-wheelers also.", f"{t.i(bikes)} బైకులకి కూడా ప్లేస్ ఉంది.",
                       f"{t.i(bikes)} दोपहिया की जगह भी है।"))
    return out


def _choice(t: T, f: Dict, c: Dict) -> List[str]:
    if c.get("lift"):
        return [t.p("Which floor shall we see? By the lift, or by the stairs? Press a button below.",
                    "ఏ ఫ్లోర్ చూద్దాం? లిఫ్ట్ లో వెళ్దామా, మెట్ల మీద వెళ్దామా? కింద బటన్ నొక్కండి.",
                    "कौन सा फ़्लोर देखें? लिफ़्ट से चलें या सीढ़ियों से? नीचे बटन दबाइए।")]
    return [t.p("Which floor shall we see? The stairs are right here. Press a button below.",
                "ఏ ఫ్లోర్ చూద్దాం? మెట్లు ఇక్కడే ఉన్నాయి. కింద బటన్ నొక్కండి.",
                "कौन सा फ़्लोर देखें? सीढ़ियाँ यहीं हैं। नीचे बटन दबाइए।")]


def _home_name(t: T, L: int, c: Dict) -> str:
    """'Flat 301' / 'the first-floor home' in this language."""
    home = c.get("homes_by_level", {}).get(L, "")
    if home.startswith("Flat "):
        no = home[5:]
        return t.p(f"Flat {t.digits(no)}", f"ఫ్లాట్ {t.digits(no)}", f"फ़्लैट {t.digits(no)}")
    fl = t.floor(L)
    return t.p(f"{fl} home", f"{fl} ఇల్లు", f"{fl} वाला घर")


def _stair_steps(t: T, c: Dict) -> Optional[str]:
    st = c.get("stair")
    if not st:
        return None
    return t.p(f"{t.i(st['steps'])} steps to every floor, with a landing half-way. Each step is only {t.inch(st['riser_mm'])} inches high — easy on the knees.",
               f"ఒక్కో ఫ్లోర్ కి {t.i(st['steps'])} మెట్లు, మధ్యలో ల్యాండింగ్. ఒక్కో మెట్టు {t.inch(st['riser_mm'])} అంగుళాలే — మోకాళ్ళకి ఈజీ.",
               f"हर फ़्लोर तक {t.i(st['steps'])} सीढ़ियाँ, बीच में लैंडिंग। हर सीढ़ी सिर्फ़ {t.inch(st['riser_mm'])} इंच ऊँची — घुटनों के लिए आसान।")


def _ride(t: T, f: Dict, c: Dict, L: int, via: str) -> List[str]:
    fl = t.floor(L)
    if L == 0 and not c.get("stilt"):                  # the ground-floor home is right here, off the lobby
        return [t.p("The ground floor home is right here, off the lobby.",
                    "గ్రౌండ్ ఫ్లోర్ ఇల్లు ఇక్కడే — లాబీ పక్కనే.",
                    "ग्राउंड फ़्लोर का घर यहीं है — लॉबी के बगल में।")]
    if via == "lift":
        return [t.p(f"Here we are — {fl}. The lift opens right into the lobby.",
                    f"{fl} కి వచ్చేసాం. లిఫ్ట్ తెరుచుకోగానే నేరుగా లాబీ.",
                    f"{fl} आ गए। लिफ़्ट खुलते ही सीधे लॉबी।")]
    out = [t.p("We go up by the stairs, flight by flight.", "మెట్ల మీద ఎక్కుతున్నాం — ఒక్కో ఫ్లైట్.",
               "सीढ़ियों से ऊपर चलते हैं — एक-एक फ़्लाइट।")]
    s = _stair_steps(t, c)
    if s:
        out.append(s)
    out.append(t.p(f"Here we are — {fl}.", f"{fl} కి వచ్చేసాం.", f"{fl} आ गए।"))
    return out


def _front_door(t: T, f: Dict, c: Dict, L: int) -> List[str]:
    home = _home_name(t, L, c)
    flats = int(c.get("flats") or 1)
    out = [t.p(f"This is the main door of {home}.", f"ఇదే {home} మెయిన్ డోర్.", f"ये {home} का मेन डोर है।")]
    if c["kind"] == "apartment" and flats > 1:
        out.append(t.p(f"{t.i(flats)} flats on this floor — all the same plan.",
                       f"ఈ ఫ్లోర్ లో {t.i(flats)} ఫ్లాట్లు — అన్నీ ఒకే ప్లాన్.",
                       f"इस फ़्लोर पर {t.i(flats)} फ़्लैट — सबका एक ही प्लान।"))
    else:
        out.append(t.p("Every floor has the same home.", "ప్రతి ఫ్లోర్ లో ఇదే ఇల్లు.", "हर फ़्लोर पर यही घर है।"))
    out.append(t.p("Come inside.", "రండి లోపలికి.", "अंदर आइए।"))
    return out


def _stair(t: T, f: Dict, c: Dict) -> List[str]:
    if f.get("external"):
        return _stair_outside(t, f, c)
    up = t.floor(1) if f.get("up") == "floor" else t.p("the terrace", "టెర్రస్", "छत")
    return [C("flight", t.p(f"The stairs go up to {up}: {t.i(f['steps'])} steps, each only {t.inch(f['riser_mm'])} inches high and "
                f"{t.inch(f['tread_mm'])} inches deep — easy even for elders.",
                f"మెట్లు {up} కి వెళ్తాయి — {t.i(f['steps'])} మెట్లు, ఒక్కో మెట్టు {t.inch(f['riser_mm'])} అంగుళాలే, "
                f"{t.inch(f['tread_mm'])} అంగుళాల వెడల్పు — పెద్దవాళ్ళకి కూడా ఎక్కడం ఈజీ.",
                f"सीढ़ियाँ {up} तक जाती हैं — {t.i(f['steps'])} सीढ़ियाँ, हर एक सिर्फ़ {t.inch(f['riser_mm'])} इंच ऊँची, "
                f"{t.inch(f['tread_mm'])} इंच चौड़ी — बुज़ुर्गों के लिए भी आसान।"))]


def _stair_outside(t: T, f: Dict, c: Dict) -> List[str]:
    """The outside stair of a one-storey house: where it stands, its Vastu, its climb, its steps — then up."""
    side = DIR[t.lang].get(f.get("compass", ""), "")
    out = [C("flight", t.p(f"This is the stair to the terrace — outside the house, on the {side} side.",
                           f"ఇవే టెర్రస్ మెట్లు — ఇంటి బయట, {side} వైపు.",
                           f"ये छत की सीढ़ियाँ हैं — घर के बाहर, {side} की तरफ़।"))]
    zone, score = f.get("zone", ""), f.get("score")
    loc = LOC[t.lang].get(zone, "")
    if score is not None and loc:
        if score >= 5:
            out.append(C("flight", t.p(f"It stands {loc} — in Vastu, a good place for a heavy stair.",
                                       f"ఇది {loc} ఉంది — వాస్తు ప్రకారం బరువైన మెట్లకి మంచి చోటు.",
                                       f"ये {loc} है — वास्तु में भारी सीढ़ियों के लिए अच्छी जगह।")))
        elif score >= 0:
            out.append(C("flight", t.p(f"It stands {loc} — Vastu-wise this is fine.",
                                       f"ఇది {loc} ఉంది — వాస్తు పరంగా పర్వాలేదు.",
                                       f"ये {loc} है — वास्तु के हिसाब से ठीक है।")))
        else:
            out.append(C("flight", t.p("Outside the house, so it never takes a single foot from your rooms.",
                                       "ఇంటి బయట ఉంది కాబట్టి — లోపల గదుల్లో ఒక్క అడుగు స్థలం కూడా పోదు.",
                                       "घर के बाहर है, तो कमरों की एक इंच जगह भी नहीं लेती।")))
    climb = f.get("climb")
    if f.get("climb_ok") and climb:
        cw = DIR[t.lang].get(climb, "")
        out.append(C("top", t.p(f"It climbs clockwise — rising towards the {cw}, just as Vastu says.",
                                f"సవ్య దిశలో ఎక్కుతాం — {cw} వైపు పైకి, వాస్తు చెప్పినట్టే.",
                                f"घड़ी की दिशा में चढ़ती है — {cw} की तरफ़ ऊपर, जैसा वास्तु कहता है।")))
    out.append(C("flight", t.p(f"{cap(t.i(f['steps']))} steps, each only {t.inch(f['riser_mm'])} inches high — easy even for elders.",
                               f"{t.i(f['steps'])} మెట్లు, ఒక్కో మెట్టు {t.inch(f['riser_mm'])} అంగుళాలే — పెద్దవాళ్ళకి కూడా ఈజీ.",
                               f"{t.i(f['steps'])} सीढ़ियाँ, हर एक सिर्फ़ {t.inch(f['riser_mm'])} इंच ऊँची — बुज़ुर्गों के लिए भी आसान।")))
    out.append(C("top", t.p("Come, let us go up.", "రండి, పైకి వెళ్దాం.", "चलिए, ऊपर चलते हैं।")))
    return out


def _terrace(t: T, f: Dict, c: Dict) -> List[str]:
    """On the roof: the space, where the tank and the solar panels go, and the view — the camera follows."""
    sq = f.get("area", 0) * SQFT
    views = set(f.get("views") or [])
    out = [C("parapet", t.p(f"And this is the terrace — {t.i(sq)} square feet open to the sky, with a one metre wall all round, safe for children.",
                            f"ఇదే టెర్రస్ — {t.i(sq)} స్క్వేర్ ఫీట్ ఓపెన్ స్పేస్, చుట్టూ ఒక మీటర్ ఎత్తు గోడ — పిల్లలకి సేఫ్.",
                            f"और ये है छत — {t.i(sq)} स्क्वेयर फ़ीट खुला आसमान, चारों तरफ़ एक मीटर ऊँची दीवार — बच्चों के लिए सुरक्षित।"))]
    out.append(C("tank", t.p("Keep the water tank on that corner — the South-West. Vastu says the highest, heaviest thing goes there.",
                             "వాటర్ ట్యాంక్ ఆ మూలలో పెట్టండి — నైరుతి. ఎత్తైనది, బరువైనది అక్కడే ఉండాలని వాస్తు చెప్తది.",
                             "पानी की टंकी उस कोने में रखिए — नैऋत्य कोण। वास्तु कहता है सबसे ऊँची, भारी चीज़ वहीं हो।")))
    out.append(C("solar", t.p("Solar panels go along the South side — full sun all day, and a smaller current bill.",
                              "సోలార్ ప్యానెల్స్ దక్షిణం వైపు — రోజంతా ఎండ, కరెంట్ బిల్లు తక్కువ.",
                              "सोलर पैनल दक्षिण की तरफ़ — दिन भर धूप, बिजली का बिल कम।")))
    if "garden" in views:
        out.append(C("garden", t.p("Look down — there is your garden.", "కిందకి చూడండి — అదే మీ గార్డెన్.",
                                   "नीचे देखिए — वो रहा आपका बगीचा।")))
    if "sump" in views:
        out.append(C("sump", t.p("And there, the water sump under the ground.", "అక్కడే భూమి కింద వాటర్ సంప్.",
                                 "और वहाँ ज़मीन के नीचे पानी का संप।")))
    if c.get("kind") in ("apartment", "floors"):
        out.append(C("road", t.p("From up here you see the whole street.", "పై నుంచి వీధి అంతా కనిపిస్తది.",
                                 "ऊपर से पूरी गली दिखती है।")))
    elif "gate" in views:
        out.append(C("gate", t.p("And there — the gate and the road.", "అటు చూడండి — గేట్, రోడ్డు.",
                                 "और वहाँ — गेट और सड़क।")))
    out.append(C("east", t.p("Every morning the sun comes up from there — the East.",
                             "రోజూ పొద్దున్నే సూర్యుడు అటు నుంచే వస్తాడు — తూర్పు.",
                             "हर सुबह सूरज वहीं से निकलता है — पूर्व।")))
    out.append(C("parapet", t.p("Drying clothes, sitting in the evening breeze — all up here.",
                                "బట్టలు ఆరేసుకోవడానికి, సాయంత్రం చల్లగాలిలో కూర్చోవడానికి — అన్నిటికీ ఇదే.",
                                "कपड़े सुखाना, शाम की ठंडी हवा में बैठना — सब यहीं।")))
    return out


def _yard(t: T, f: Dict, c: Dict) -> List[str]:
    """Around the house before going in: the garden, the sump, the pool, the generator, the wall — one look each."""
    out = [C("house", t.p("Before we go inside — see the open space around the house.",
                          "లోపలికి వెళ్ళే ముందు — ఇంటి చుట్టూ ఖాళీ స్థలం చూడండి.",
                          "अंदर जाने से पहले — घर के चारों तरफ़ की खुली जगह देखिए।"))]
    for it in f.get("items") or []:
        k, zone = it["kind"], it.get("zone", "")
        loc = LOC[t.lang].get(zone, "")
        if k == "GARDEN":
            good = zone in ("N", "NE", "E")
            out.append(C("garden", t.p(f"The garden is {loc}." + (" Vastu likes the North and East open and green — so it is right there." if good
                                                                 else " Green and cool — a place to sit in the evening."),
                                       f"గార్డెన్ {loc} ఉంది." + (" ఉత్తరం, తూర్పు ఖాళీగా, పచ్చగా ఉండాలని వాస్తు చెప్తది — అందుకే అక్కడే." if good
                                                                   else " పచ్చగా, చల్లగా — సాయంత్రం కూర్చోవడానికి బాగుంటది."),
                                       f"बगीचा {loc} है।" + (" वास्तु कहता है उत्तर और पूर्व खुले और हरे रहें — इसलिए वहीं है।" if good
                                                              else " हरा-भरा, ठंडा — शाम को बैठने की जगह।"))))
        elif k == "WATER_TANK":
            ne = zone == "NE"
            out.append(C("sump", t.p(f"The underground water sump is {loc}." + (" Water below the ground in the North-East — the best in Vastu." if ne else ""),
                                     f"అండర్ గ్రౌండ్ వాటర్ సంప్ {loc} ఉంది." + (" ఈశాన్యంలో భూమి కింద నీళ్ళు — వాస్తు ప్రకారం చాలా మంచిది." if ne else ""),
                                     f"अंडरग्राउंड पानी का संप {loc} है।" + (" ईशान कोण में ज़मीन के नीचे पानी — वास्तु में सबसे अच्छा।" if ne else ""))))
        elif k == "POOL":
            out.append(C("pool", t.p(f"And the swimming pool, {loc}.", f"స్విమ్మింగ్ పూల్ {loc}.", f"और स्विमिंग पूल, {loc}।")))
        elif k == "GENERATOR":
            out.append(C("genset", t.p("The generator stands here, away from the bedrooms — power cuts are no worry.",
                                       "జనరేటర్ ఇక్కడ — బెడ్రూమ్స్ కి దూరంగా; కరెంట్ పోయినా ఇబ్బంది లేదు.",
                                       "जनरेटर यहाँ है — बेडरूम से दूर; बिजली जाए तो भी चिंता नहीं।")))
        elif k == "SECURITY":
            out.append(C("security", t.p("A security cabin near the gate.", "గేట్ దగ్గర సెక్యూరిటీ క్యాబిన్.",
                                         "गेट के पास सिक्योरिटी केबिन।")))
    ft = f.get("compound_ft")
    if ft:
        out.append(C("compound", t.p(f"A {t.i(ft)} feet compound wall goes all round — safe, and private.",
                                     f"చుట్టూ {t.i(ft)} అడుగుల కాంపౌండ్ వాల్ — సేఫ్టీ, ప్రైవసీ రెండూ.",
                                     f"चारों तरफ़ {t.i(ft)} फ़ीट की बाउंड्री वॉल — सुरक्षा भी, प्राइवेसी भी।")))
    return out


def _more(t: T, f: Dict, c: Dict, L: int, nxt: Optional[int]) -> List[str]:
    """After a floor's home: on to the next floor, or straight up to the terrace?"""
    home = _home_name(t, L, c)
    if nxt is not None:
        nf = t.floor(nxt)
        out = [C("lift", t.p(f"That was {home}. Shall we go up to {nf}? Or straight up to the terrace?",
                             f"ఇదీ {home}. {nf} కి వెళ్దామా? లేక నేరుగా టెర్రస్ కి వెళ్దామా?",
                             f"ये था {home}। {nf} चलें? या सीधे छत पर?"))]
    else:
        out = [C("lift", t.p(f"That was {home} — the top floor. Shall we go up to the terrace?",
                             f"ఇదీ {home} — ఇదే పై ఫ్లోర్. టెర్రస్ కి వెళ్దామా?",
                             f"ये था {home} — सबसे ऊपर का फ़्लोर। छत पर चलें?"))]
    out.append(t.p("Press a button below.", "కింద బటన్ నొక్కండి.", "नीचे बटन दबाइए।"))
    return out


def _more_options(t: T, c: Dict, nxt: Optional[int]) -> List[Dict]:
    """The buttons of the 'more' stop (captions: digits are fine)."""
    opts = []
    vias = ["lift", "stairs"] if c.get("lift") else ["stairs"]
    if nxt is not None:
        nf = cap(t.floor(nxt).replace("the ", ""))
        for v in vias:
            how = t.p(" — by the lift", " — లిఫ్ట్ లో", " — लिफ़्ट से") if v == "lift" else \
                t.p(" — by the stairs", " — మెట్ల మీద", " — सीढ़ियों से")
            opts.append({"act": "next", "L": nxt, "via": v, "label": ("🛗 " if v == "lift" else "🪜 ") + nf + how})
    opts.append({"act": "terrace", "via": vias[0],
                 "label": t.p("🏙️ Straight to the terrace", "🏙️ నేరుగా టెర్రస్ కి", "🏙️ सीधे छत पर")})
    opts.append({"act": "end", "label": t.p("⏹ That's enough — finish", "⏹ చాలు — ముగిద్దాం", "⏹ बस — यहीं ख़त्म")})
    return opts


def _room_again(t: T, f: Dict, c: Dict) -> List[str]:
    """A room on the next floor up: the same plan — said in a line or two, not all over again."""
    kind = f["kind"]
    name = _tr(t, ROOM.get(kind, ROOM["ROOM"]))
    w, h, sqft = f.get("w", 0), f.get("h", 0), f.get("sqft", 0)
    size = t.p(f"{t.ft(w)} by {t.ft(h)} feet", f"{t.ft(w)} బై {t.ft(h)} అడుగులు", f"{t.ft(w)} बाई {t.ft(h)} फ़ीट")
    if kind in ("LIVING", "FAMILY"):
        return [C("room", t.p(f"The same {name} as below — {size}. Up here, more breeze, less dust and street noise.",
                              f"కింద ఉన్న {name} లాగే — {size}. పైన గాలి ఎక్కువ, దుమ్ము, రోడ్డు శబ్దం తక్కువ.",
                              f"नीचे जैसा ही {name} — {size}। ऊपर हवा ज़्यादा, धूल और सड़क का शोर कम।"))]
    if kind.endswith("BEDROOM"):
        out = [C("room", t.p(f"{cap(name)} — the same plan as below, {size}.", f"{name} — కింద లాగే, {size}.",
                             f"{name} — नीचे जैसा ही, {size}।"))]
        bed = (f.get("fits") or {}).get("bed")
        if bed:
            b = _tr(t, _BED[bed])
            out.append(C("bed", t.p(f"{cap(b)} bed fits here too.", f"{b} బెడ్ ఇక్కడా పడతది.", f"{b} बेड यहाँ भी आ जाएगा।")))
        return out
    if kind == "KITCHEN":
        return [C("counter", t.p("Kitchen — the same as below, everything in its place.",
                                 "కిచెన్ — కింద లాగే, అన్నీ అదే ప్లేస్ లో.", "किचन — नीचे जैसा ही, सब कुछ अपनी जगह।"))]
    return [C("room", t.p(f"{cap(name)} — same as below.", f"{name} — కింద లాగే.", f"{name} — नीचे जैसा ही।"))]


def _end(t: T, f: Dict, c: Dict, L: Optional[int] = None) -> List[str]:
    out = []
    if c["kind"] in ("apartment", "floors") and L is not None:
        seen = [t.p("the gate", "గేట్", "गेट")]
        if c.get("seen_parking"):
            seen.append(t.p("the parking", "పార్కింగ్", "पार्किंग"))
        seen.append(t.p("the lift and stairs", "లిఫ్ట్, మెట్లు", "लिफ़्ट और सीढ़ियाँ") if c.get("lift")
                    else t.p("the stairs", "మెట్లు", "सीढ़ियाँ"))
        seen.append(_home_name(t, L, c))
        if c.get("seen_terrace"):
            seen.append(t.p("the terrace", "టెర్రస్", "छत"))
        out.append(t.p(f"That is the whole building — {t.join(seen)}.", f"ఇదీ అండీ మొత్తం బిల్డింగ్ — {t.join(seen)}.",
                       f"ये रही पूरी बिल्डिंग — {t.join(seen)}।"))
    else:
        out.append(t.p("That is the whole house.", "ఇదీ అండీ మొత్తం ఇల్లు.", "ये रहा पूरा घर।"))
    hl = _home_highlights(t, c)
    if hl:
        out.append(hl)
    vs = c.get("vastu")
    if vs is not None and vs >= 70:
        word = t.p("very good", "చాలా మంచి స్కోర్", "बहुत बढ़िया") if vs >= 85 else t.p("good", "మంచి స్కోర్", "अच्छा")
        out.append(t.p(f"Vastu score {t.i(vs, False)} out of hundred — {word}.",
                       f"వాస్తు స్కోర్ వందకి {t.i(vs, False)} — {word}.",
                       f"वास्तु स्कोर सौ में से {t.i(vs, False)} — {word}।"))
    low = int(c.get("nbc_low") or 0)
    if not low:
        out.append(t.p("Every room is as per the National Building Code sizes.",
                       "ప్రతి గది నేషనల్ బిల్డింగ్ కోడ్ సైజుల ప్రకారమే.",
                       "हर कमरा नेशनल बिल्डिंग कोड के साइज़ के हिसाब से है।"))
    cols = int(c.get("columns") or 0)
    if cols:
        out.append(t.p(f"Strong RCC frame on {t.i(cols)} columns, outer walls nine inches thick.",
                       f"{t.i(cols)} ఆర్సీసీ కాలమ్స్ మీద స్ట్రాంగ్ బిల్డింగ్, బయటి గోడలు తొమ్మిది అంగుళాలు.",
                       f"{t.i(cols)} आरसीसी कॉलम पर मज़बूत ढाँचा, बाहर की दीवारें नौ इंच मोटी।"))
    extras = [w for k, w in (("garden", t.p("a garden", "గార్డెన్", "बगीचा")), ("pool", t.p("a swimming pool", "స్విమ్మింగ్ పూల్", "स्विमिंग पूल")))
              if c.get(k)]
    if extras:
        out.append(t.p(f"And {t.join(extras)} also.", f"{t.join(extras)} కూడా ఉంది.", f"और {t.join(extras)} भी है।"))
    m = int(c.get("members") or 0)
    if m:
        pp = c.get("carpet_m2", 0) / max(1, m)
        if pp >= 11:
            out.append(t.p(f"For your family of {t.i(m)}, it fits just right.",
                           f"మీ {t.i(m)} మంది ఫ్యామిలీకి సరిగ్గా సరిపోతది.",
                           f"आपके {t.i(m)} लोगों के परिवार के लिए एकदम सही।"))
        else:
            out.append(t.p(f"A cosy home for your family of {t.i(m)} — every room in use, nothing wasted.",
                           f"మీ {t.i(m)} మంది ఫ్యామిలీకి ముద్దుగా సరిపోయే ఇల్లు — ప్రతి గది వాడుకలోనే, ఏదీ వృథా కాదు.",
                           f"आपके {t.i(m)} लोगों के परिवार के लिए प्यारा-सा घर — हर कमरा काम का, कुछ भी बेकार नहीं।"))
    out.append(t.p("Did you like it? To change any room, stand in it and press Change this room.",
                   "నచ్చిందా అండీ? ఏ గదైనా మార్చాలంటే, ఆ గదిలో నిలబడి చేంజ్ బటన్ నొక్కండి.",
                   "पसंद आया? कोई कमरा बदलना हो तो उसमें खड़े होकर चेंज बटन दबाइए।"))
    return out


# =============================================================================
# Assembly
# =============================================================================

def key(lang: str, text: str) -> str:
    """The voice clip's name for a sentence."""
    return hashlib.sha1(f"{lang}|{text}".encode("utf-8")).hexdigest()[:16]


def _clean_speech(s: str) -> str:
    s = s.replace(" — ", ", ").replace("—", ", ").replace("·", ",").replace("  ", " ")
    s = re.sub(r"[←-⇿☀-➿\U0001F300-\U0001FAFF]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _render(fn, lang: str, *args) -> Dict:
    """Caption + speech of one stop; each spoken line keeps its cue ('c') and its caption sentence ('lines')."""
    cap = fn(T(lang, False), *args)
    spk = fn(T(lang, True), *args)
    speech, lines = [], []
    for i, s in enumerate(spk):
        cue = getattr(s, "cue", None)
        s = _clean_speech(s)
        if s:
            line = {"k": key(lang, s), "t": s}
            if cue:
                line["c"] = cue
            speech.append(line)
            lines.append(str(cap[i]) if i < len(cap) else "")
    return {"say": " ".join(cap), "speech": speech, "lines": lines}


def _aim(out: Dict, cues: Optional[Dict]) -> None:
    """Put each cued line's point on it ('at'), from the stop's measured cues; unknown cues just look ahead."""
    if not out:
        return
    groups = [out.get("speech") or []] + [v.get("speech") or [] for v in (out.get("by") or {}).values()] + \
        [(out.get("again") or {}).get("speech") or []]
    for g in groups:
        for line in g:
            c = line.get("c")
            if c and cues and c in cues:
                line["at"] = cues[c]


_TITLES = {
    "start": ("Welcome", "స్వాగతం", "स्वागत है"),
    "gate": ("The gate", "గేట్", "गेट"),
    "parking": ("Parking", "పార్కింగ్", "पार्किंग"),
    "main_door": ("Main door", "మెయిన్ డోర్", "मेन डोर"),
    "entrance": ("Building entrance", "బిల్డింగ్ ఎంట్రన్స్", "बिल्डिंग एंट्रेंस"),
    "core": ("Stairs and lift", "మెట్లు, లిఫ్ట్", "सीढ़ियाँ और लिफ़्ट"),
    "core_stairs": ("Stairs", "మెట్లు", "सीढ़ियाँ"),
    "choice": ("Lift or stairs?", "లిఫ్టా? మెట్లా?", "लिफ़्ट या सीढ़ियाँ?"),
    "choice_stairs": ("Which floor?", "ఏ ఫ్లోర్?", "कौन सा फ़्लोर?"),
    "stair": ("Staircase", "మెట్లు", "सीढ़ियाँ"),
    "stair_out": ("Terrace stair", "టెర్రస్ మెట్లు", "छत की सीढ़ियाँ"),
    "terrace": ("Terrace", "టెర్రస్", "छत"),
    "yard": ("Around the house", "ఇంటి చుట్టూ", "घर के चारों तरफ़"),
    "more": ("Next floor?", "తర్వాతి ఫ్లోర్?", "अगला फ़्लोर?"),
    "end": ("That's your home", "ఇదీ మీ ఇల్లు", "ये रहा आपका घर"),
}
VIA = {"lift": ("🛗 By the lift", "🛗 లిఫ్ట్ లో", "🛗 लिफ़्ट से"), "stairs": ("🪜 By the stairs", "🪜 మెట్ల మీద", "🪜 सीढ़ियों से")}


def _title(lang: str, k: str) -> str:
    return _TITLES[k][LANGS.index(lang)]


def add_talk(tour: List[Dict], ctx: Dict) -> List[Dict]:
    """Give every stop its broker's words in en / te / hi: stop['talk'][lang] = {title, say, speech[, by, options, via]}."""
    choice = next((s for s in tour if s.get("kind") == "choice"), None)
    levels = [o["L"] for o in (choice or {}).get("options", [])]
    vias = (choice or {}).get("via") or ["stairs"]
    ctx = dict(ctx)
    ctx["homes_by_level"] = {o["L"]: o.get("home", "") for o in (choice or {}).get("options", [])}
    ctx["seen_parking"] = any(s.get("kind") == "parking" for s in tour)
    ctx["seen_terrace"] = any(s.get("kind") == "terrace" for s in tour)
    ctx["gate_stop"] = any(s.get("kind") == "gate" for s in tour)
    door = next((s for s in tour if s.get("kind") == "door" and (s.get("facts") or {}).get("door") == "main"), None)
    ctx["door_lucky"] = bool(door and ((door.get("facts") or {}).get("score") or -99) >= 8)
    # the highlights, room by room in walking order: a highlight said before weighs less the next time
    used: Dict[str, int] = {}
    home = {"door": ctx["door_lucky"], "cross": 0, "baths": 0, "east": 0}
    rooms = [s for s in tour if s.get("kind") == "room"]
    for i, st in enumerate(rooms):
        f = st.setdefault("facts", {})
        hl = room_highlights(dict(f, lift=bool(ctx.get("lift"))), used)
        for pid, a in hl:
            a["_n"] = used.get(pid, 0)
            used[pid] = used.get(pid, 0) + 1
        f["_hl"] = hl
        kind, z = f.get("kind", ""), f.get("zone")
        if kind == "KITCHEN" and z == "SE":
            home["kitchen_se"] = True
        if kind == "MASTER_BEDROOM" and z == "SW":
            home["master_sw"] = True
        if kind == "POOJA" and z == "NE":
            home["pooja_ne"] = True
        wins = list(dict.fromkeys(f.get("wins") or []))
        home["cross"] += len(wins) >= 2
        home["east"] += "E" in wins
        home["baths"] += kind == "ATTACHED_BATH"
        # a bedroom whose own bathroom is the next stop or two leads you into it
        nxt = rooms[i + 1:i + 3]
        f["bath_next"] = any((q.get("facts") or {}).get("kind") == "ATTACHED_BATH" for q in nxt) and kind.endswith("BEDROOM")
        if f["bath_next"] and f.get("attached"):
            f["tease_n"] = used.get("_tease", 0)
            used["_tease"] = used.get("_tease", 0) + 1
    ctx["home_hl"] = home
    for st in tour:
        kind, f = st.get("kind"), st.get("facts") or {}
        talk = {}
        for lang in LANGS:
            t0 = T(lang, False)
            by = None
            if kind == "start":
                out, title = _render(lambda t, *a: _tag(_start(t, f, ctx), "house", "house"), lang), _title(lang, "start")
            elif kind == "gate":
                g = f.get("gate") or {}
                g0 = "gate_car" if g.get("car_ft") else "gate_walk"
                out = _render(lambda t, *a: _tag(_gate_lines(t, g), g0, g0, "gate_walk",
                                                 last_if=bool(g.get("car_ft") and g.get("walk_ft") and not g.get("wicket"))),
                              lang)
                title = _title(lang, "gate")
            elif kind == "parking":
                out = _render(lambda t, *a: _tag(_parking(t, f, ctx), "car", "car", "bikes", last_if=bool(f.get("bikes"))),
                              lang)
                title = _title(lang, "parking")
            elif kind == "yard":
                out, title = _render(_yard, lang, f, ctx), _title(lang, "yard")
            elif kind == "door" and f.get("door") == "main":
                out = _render(lambda t, *a: _tag(_main_door(t, f, ctx), "door", "door", "steps",
                                                 last_if=bool(ctx.get("plinth_cm"))), lang)
                title = _title(lang, "main_door")
            elif kind == "door" and f.get("door") in ("entrance", "core"):
                out = _render(lambda t, *a: _tag(_entrance(t, f, ctx), "stair" if f.get("stilt") else "lobby", "lift"), lang)
                title = _title(lang, "entrance" if f.get("door") == "entrance" else
                               ("core" if ctx.get("lift") else "core_stairs"))
            elif kind == "door" and f.get("door") == "front":
                out = {"say": "", "speech": []}
                title = ""
                by = {}
                for L in levels:
                    r = _render(lambda t, *a: _tag(_front_door(t, f, ctx, L), "door", "door"), lang)
                    r["title"] = T(lang, False).p("Front door — ", "మెయిన్ డోర్ — ", "मेन डोर — ") + _home_name(T(lang, False), L, ctx)
                    by[str(L)] = r
            elif kind == "choice":
                out = _render(lambda t, *a: _tag(_choice(t, f, ctx), "lift" if ctx.get("lift") else "stair"), lang)
                title = _title(lang, "choice" if ctx.get("lift") else "choice_stairs")
                out["options"] = []
                for o in choice.get("options", []):
                    fl = t0.floor(o["L"]).replace("the ", "")
                    label = fl[:1].upper() + fl[1:]
                    if ctx["kind"] == "apartment":
                        label += " · " + _home_name(t0, o["L"], ctx)
                    out["options"].append({"L": o["L"], "label": label})
                out["via"] = {v: VIA[v][LANGS.index(lang)] for v in vias}
            elif kind == "ride":
                out, title = {"say": "", "speech": []}, ""
                by = {}
                for L in levels:
                    for v in vias:
                        r = _render(lambda t, *a: _tag(_ride(t, f, ctx, L, v), "lobby", "lobby"), lang)
                        fl = t0.floor(L).replace("the ", "")
                        r["title"] = cap(fl) if (L == 0 and not ctx.get("stilt")) else                             t0.p("Going up to ", "పైకి — ", "ऊपर — ") + fl
                        by[f"{L}|{v}"] = r
            elif kind == "room":
                rf = dict(f)
                out = _render(_room, lang, rf, ctx)
                title = _tr(t0, ROOM.get(rf.get("kind", "ROOM"), ROOM["ROOM"]))
                title = title[:1].upper() + title[1:] if lang == "en" else title
                if ctx["kind"] in ("apartment", "floors"):          # the same room on the next floor up
                    out["again"] = _render(_room_again, lang, rf, ctx)
                    out["again"]["title"] = title
            elif kind == "stair":
                out = _render(_stair, lang, f, ctx)
                title = _title(lang, "stair_out" if f.get("external") else "stair")
            elif kind == "more":
                out, title = {"say": "", "speech": []}, _title(lang, "more")
                by = {}
                for L in levels:
                    nxt = next((x for x in levels if x > L), None)
                    r = _render(lambda t, *a: _more(t, f, ctx, L, nxt), lang)
                    r["title"] = _title(lang, "more") if nxt is not None else _title(lang, "terrace")
                    r["options"] = _more_options(t0, ctx, nxt)
                    by[str(L)] = r
            elif kind == "terrace":
                out, title = _render(_terrace, lang, f, ctx), _title(lang, "terrace")
            elif kind == "end":
                if ctx["kind"] in ("apartment", "floors") and levels:
                    out, title = {"say": "", "speech": []}, _title(lang, "end")
                    by = {}
                    for L in levels:
                        r = _render(lambda t, *a: _end(t, f, ctx, L), lang)
                        r["title"] = _title(lang, "end")
                        by[str(L)] = r
                else:
                    out, title = _render(_end, lang, f, ctx), _title(lang, "end")
            else:
                continue
            out["title"] = title
            if by is not None:
                out["by"] = by
            _aim(out, st.get("cues"))
            talk[lang] = out
        if talk:
            st["talk"] = talk
    return tour


def speech_lines(tour: List[Dict], lang: str) -> List[Dict]:
    """Every sentence the guide may say in `lang`, in walking order, once each — what the voice must record."""
    seen, out = set(), []
    for st in tour:
        tk = (st.get("talk") or {}).get(lang)
        if not tk:
            continue
        groups = [tk.get("speech", [])] + [v.get("speech", []) for v in (tk.get("by") or {}).values()] + \
            [(tk.get("again") or {}).get("speech", [])]
        for g in groups:
            for s in g:
                if s["k"] not in seen:
                    seen.add(s["k"])
                    out.append(s)
    return out

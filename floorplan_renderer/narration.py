"""
🗣️ The guide's words — the guided walk told the way a good broker shows a home
==============================================================================
Every stop of the 3D guided walk (floorplan_renderer/walkthrough.py) gets its words in three languages:

  en  simple Indian English — short sentences, everyday words ("hall", "car parking is also there"),
      the English an owner who learned it at school speaks and understands
  te  Telugu as it is spoken around Guntur — colloquial (వస్తది, ఉంటది, అండీ), with the English words
      brokers really use (హాల్, కిచెన్, ఈస్ట్ ఫేసింగ్) and Vastu's own directions (ఈశాన్యం, ఆగ్నేయం…)
  hi  natural spoken Hindi — जी, देखिए, एकदम; Vastu's directions (ईशान कोण, आग्नेय कोण…)

Like a broker it sells — the morning sun, the Vastu corner, the king-size bed, the rent from the other
floors — but every claim is measured from the plan (sizes, Vastu zones and scores, windows, furniture,
gates, stairs, lift) and a weak point is said too, softly, with what can be done about it.

Each line is made twice from the same template: the caption (digits, short units) and the speech
(numbers in words, no symbols) as a list of sentences. A sentence's key k = sha1(lang|text)[:16] names its
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

def _vastu_line(t: T, kind: str, f: Dict) -> Optional[str]:
    """What Vastu says about this room where it is — the broker's favourite line, honest when it is weak."""
    z, s = f.get("zone"), f.get("score")
    if not z or s is None:
        return None
    loc = LOC[t.lang].get(z, "")
    if kind == "POOJA" and z == "NE":
        return t.p("North-East is God's corner in Vastu — there is no better place for pooja.",
                   "ఈశాన్యం దేవుడి మూల — పూజకి అంతకంటే మంచి ప్లేస్ లేదు అని వాస్తు చెప్తది.",
                   "ईशान कोण भगवान का कोना है — पूजा के लिए इससे अच्छी जगह नहीं।")
    if kind == "KITCHEN" and z == "SE":
        return t.p("South-East is the fire corner — Vastu says cooking belongs here. A proper Vastu kitchen.",
                   "ఆగ్నేయం అగ్ని మూల — వంట అక్కడే చేయాలని వాస్తు చెప్తది. పక్కా వాస్తు కిచెన్.",
                   "आग्नेय कोण अग्नि का कोना है — वास्तु कहता है खाना यहीं बनना चाहिए। एकदम वास्तु वाला किचन।")
    if kind == "KITCHEN" and z == "NW":
        return t.p("Kitchen in the North-West — after South-East, Vastu's next good place for cooking.",
                   "వాయువ్యంలో కిచెన్ — ఆగ్నేయం తర్వాత వంటకి వాస్తులో రెండో మంచి ప్లేస్ ఇదే.",
                   "वायव्य कोण में किचन — आग्नेय के बाद वास्तु में खाना बनाने की दूसरी अच्छी जगह यही है।")
    if kind == "MASTER_BEDROOM" and z == "SW":
        return t.p("South-West is for the head of the family — Vastu says sleeping here keeps the home steady.",
                   "నైరుతి ఇంటి పెద్ద దిక్కు — ఇక్కడ పడుకుంటే ఇంట్లో స్థిరత్వం ఉంటది అని వాస్తు చెప్తది.",
                   "नैऋत्य कोण घर के मुखिया का है — वास्तु कहता है यहाँ सोने से घर में स्थिरता रहती है।")
    if kind in ("LIVING", "FAMILY") and z in ("N", "NE", "E") and s >= 3:
        return t.p(f"Hall {loc} — as per Vastu very good, the house feels light and open.",
                   f"హాల్ {loc} — వాస్తు ప్రకారం చాలా మంచిది, ఇల్లంతా వెలుతురు, పాజిటివ్ గా ఉంటది.",
                   f"हॉल {loc} — वास्तु के हिसाब से बहुत बढ़िया, घर खुला-खुला और रोशन रहता है।")
    if kind in ("BATHROOM", "ATTACHED_BATH", "STORE", "UTILITY", "CORRIDOR", "LOBBY", "FOYER", "BALCONY"):
        if s > -30:
            return None                                   # nobody asks; say it only when Vastu forbids it
    if s >= 8:
        return t.p(f"It is {loc} — as per Vastu, the best place for it.",
                   f"ఇది {loc} ఉంది — వాస్తు ప్రకారం దీనికి ఇదే బెస్ట్ ప్లేస్.",
                   f"ये {loc} है — वास्तु के हिसाब से इसके लिए यही सबसे अच्छी जगह है।")
    if s >= 3:
        return t.p(f"It is {loc} — a good place in Vastu also.",
                   f"ఇది {loc} ఉంది — వాస్తుకి కూడా మంచి ప్లేస్.",
                   f"ये {loc} है — वास्तु के हिसाब से भी अच्छी जगह।")
    if s >= -10:
        return t.p(f"It is {loc} — Vastu-wise this is fine.",
                   f"ఇది {loc} ఉంది — వాస్తు పరంగా పర్వాలేదు.",
                   f"ये {loc} है — वास्तु के हिसाब से ठीक है।")
    best = [DIR[t.lang].get(b, b) for b in (f.get("best") or [])[:2]]
    better = t.p(" or ".join(best), " లేదా ".join(best), " या ".join(best)) if best else ""
    here = "the " + DIR["en"].get(z, z) + (" of the house" if z == "CENTER" else "")
    if better:
        return t.p(f"Honestly, in Vastu {here} is not a good place for this room — {better} would be better. "
                   f"Smart Re-Plan can move it.",
                   f"నిజం చెప్పాలంటే, ఈ గదికి {loc} వాస్తుకి అంత మంచిది కాదు — {better} అయితే బాగుంటది. "
                   f"స్మార్ట్ రీ-ప్లాన్ లో మార్చుకోవచ్చు.",
                   f"सच कहूँ तो इस कमरे के लिए {loc} होना वास्तु में ठीक नहीं — {better} बेहतर रहेगा। "
                   f"स्मार्ट री-प्लान से बदल सकते हैं।")
    return t.p(f"Honestly, in Vastu {here} is not a good place for this room.",
               f"నిజం చెప్పాలంటే, ఈ గదికి {loc} వాస్తుకి అంత మంచిది కాదు.",
               f"सच कहूँ तो इस कमरे के लिए {loc} होना वास्तु में ठीक नहीं।")


def _sides(t: T, sides: List[str]) -> str:
    return t.join([DIR[t.lang].get(s, s) for s in sides])


def _light_line(t: T, kind: str, f: Dict) -> Optional[str]:
    wins = f.get("wins") or []
    wet = kind in ("BATHROOM", "ATTACHED_BATH", "STORE")
    if f.get("vent") and wet:
        return t.p("A ventilator high on the wall lets the air out — no smell stays inside.",
                   "పైన వెంటిలేటర్ ఉంది — గాలి బయటికి పోతది, వాసన ఉండదు.",
                   "ऊपर वेंटिलेटर है — हवा बाहर निकलती है, बदबू नहीं रुकती।")
    if not wins:
        if wet and kind != "STORE":
            return t.p("It has no outside wall, so an exhaust fan is fitted.",
                       "దీనికి బయటి గోడ లేదు, అందుకే ఎగ్జాస్ట్ ఫ్యాన్ పెడతాం.",
                       "इसकी बाहरी दीवार नहीं है, इसलिए एग्ज़ॉस्ट फ़ैन लगेगा।")
        if kind in ("LIVING", "FAMILY", "DINING", "KITCHEN", "STUDY") or kind.endswith("BEDROOM"):
            return t.p("I will tell you honestly — no outside window here, so it needs a light and a fan.",
                       "నిజం చెప్తా — ఇక్కడ బయటికి కిటికీ లేదు, లైట్, ఫ్యాన్ కావాలి.",
                       "सच बताऊँ — यहाँ बाहर की खिड़की नहीं है, लाइट और पंखा चाहिए।")
        return None
    if wet:
        return None
    uniq = list(dict.fromkeys(wins))
    if len(uniq) >= 2:
        return t.p(f"Windows on two sides — {_sides(t, uniq)}. Cross ventilation, air flows right through.",
                   f"రెండు వైపులా కిటికీలు — {_sides(t, uniq)}. క్రాస్ వెంటిలేషన్, గాలి ధారాళంగా వస్తది.",
                   f"दो तरफ़ खिड़कियाँ — {_sides(t, uniq)}। क्रॉस वेंटिलेशन, हवा आर-पार चलती है।")
    if "E" in uniq:
        return t.p("Window on the East — morning sun comes straight inside.",
                   "తూర్పు వైపు కిటికీ — పొద్దున్నే ఎండ నేరుగా లోపలికి వస్తది.",
                   "पूर्व की तरफ़ खिड़की — सुबह की धूप सीधे अंदर आती है।")
    if uniq == ["W"]:
        return t.p("The window faces West — evenings can be warm, a curtain will help.",
                   "కిటికీ పడమర వైపు — సాయంత్రం కొంచెం వేడిగా ఉంటది, కర్టెన్ పెట్టుకుంటే సరిపోతది.",
                   "खिड़की पश्चिम की तरफ़ है — शाम को थोड़ी गर्मी रहेगी, पर्दा लगा लीजिए।")
    side = DIR[t.lang].get(uniq[0], uniq[0])
    return t.p(f"Window on the {side} side — good daylight, good air.",
               f"{side} వైపు కిటికీ — మంచి వెలుతురు, మంచి గాలి.",
               f"{side} की तरफ़ खिड़की — अच्छी रोशनी, अच्छी हवा।")


_BED = {"king": ("king size", "కింగ్ సైజ్", "किंग साइज़"), "queen": ("queen size", "క్వీన్ సైజ్", "क्वीन साइज़"),
        "double": ("double", "డబుల్", "डबल"), "single": ("single", "సింగిల్", "सिंगल")}


def _fits_lines(t: T, kind: str, fits: Dict) -> List[str]:
    out = []
    bed = fits.get("bed")
    if bed:
        b = _tr(t, _BED[bed])
        extra = fits.get("wardrobe")
        if extra:
            out.append(t.p(f"{cap(b)} bed fits easily — wardrobe also.",
                           f"{b} బెడ్ సులువుగా పడతది — వార్డ్రోబ్ కూడా.",
                           f"{b} बेड आराम से आ जाएगा — वॉर्डरोब भी।"))
        else:
            out.append(t.p(f"{cap(b)} bed fits easily.", f"{b} బెడ్ సులువుగా పడతది.", f"{b} बेड आराम से आ जाएगा।"))
    if fits.get("desk") and kind != "STUDY":
        out.append(t.p("A study table also fits — a separate place to read.",
                       "స్టడీ టేబుల్ కూడా పడతది — చదువుకోవడానికి సెపరేట్ ప్లేస్.",
                       "स्टडी टेबल भी आ जाएगी — पढ़ने के लिए अलग जगह।"))
    if fits.get("sofa"):
        n = fits["sofa"]
        out.append(t.p(f"{t.i(n)} seater sofa and TV unit — even when relatives come, no problem.",
                       f"{t.i(n)} సీటర్ సోఫా, టీవీ యూనిట్ — చుట్టాలు వచ్చినా ఇరుకుగా ఉండదు.",
                       f"{t.i(n)} सीटर सोफ़ा और टीवी यूनिट — मेहमान आएँ तो भी जगह की कमी नहीं।"))
    if fits.get("dining"):
        n = fits["dining"]
        out.append(t.p(f"Dining table for {t.i(n)} — the whole family eats together.",
                       f"{t.i(n)} మంది కూర్చునే డైనింగ్ టేబుల్ — ఫ్యామిలీ అంతా కలిసి తినొచ్చు.",
                       f"{t.i(n)} लोगों की डाइनिंग टेबल — पूरा परिवार साथ बैठकर खाए।"))
    if fits.get("counter"):
        out.append(t.p(f"{t.h(fits['counter'])} metres of counter — stove, sink and fridge, all have their place.",
                       f"{t.h(fits['counter'])} మీటర్ల కౌంటర్ — స్టవ్, సింక్, ఫ్రిడ్జ్ అన్నిటికీ ప్లేస్ ఉంది.",
                       f"{t.h(fits['counter'])} मीटर का काउंटर — चूल्हा, सिंक, फ़्रिज सबकी जगह है।"))
    if fits.get("washer"):
        out.append(t.p("Place for the washing machine also.", "వాషింగ్ మెషిన్ కి కూడా ప్లేస్ ఉంది.",
                       "वॉशिंग मशीन की जगह भी है।"))
    if kind in ("BATHROOM", "ATTACHED_BATH") and (fits.get("wc") or fits.get("shower")):
        parts = [w for k, w in (("wc", t.p("toilet", "టాయిలెట్", "टॉयलेट")), ("shower", t.p("shower", "షవర్", "शावर")),
                                ("basin", t.p("wash basin", "వాష్ బేసిన్", "वॉश बेसिन"))) if fits.get(k)]
        out.append(t.p(f"Inside: {t.join(parts)}.", f"లోపల {t.join(parts)}.", f"अंदर {t.join(parts)}।"))
    return out


def _room(t: T, f: Dict, c: Dict) -> List[str]:
    kind = f["kind"]
    name = _tr(t, ROOM.get(kind, ROOM["ROOM"]))
    w, h, sqft = f.get("w", 0), f.get("h", 0), f.get("sqft", 0)
    big = (kind in ("LIVING", "FAMILY") and f.get("area", 0) >= 14) or \
          (kind.endswith("BEDROOM") and f.get("area", 0) >= 12.5)
    size = t.p(f"{t.ft(w)} by {t.ft(h)} feet, {t.i(sqft)} square feet",
               f"{t.ft(w)} బై {t.ft(h)} అడుగులు, {t.i(sqft)} స్క్వేర్ ఫీట్",
               f"{t.ft(w)} बाई {t.ft(h)} फ़ीट, {t.i(sqft)} स्क्वेयर फ़ीट")
    out = []
    if f.get("first_up"):
        out.append(t.p("Now we are upstairs.", "ఇప్పుడు పైకి వచ్చాం.", "अब हम ऊपर आ गए।"))
    if kind in ("LIVING", "FAMILY"):
        out.append(t.p(f"See, this is the {name} — {size}." + (" Nice and big." if big else ""),
                       f"చూడండి, ఇదే {name} — {size}." + (" బాగా పెద్దది." if big else ""),
                       f"देखिए, ये रहा {name} — {size}।" + (" खुला-खुला, बड़ा।" if big else "")))
    elif kind == "MASTER_BEDROOM":
        out.append(t.p(f"This is the master bedroom, for you both — {size}.",
                       f"ఇదే మాస్టర్ బెడ్రూమ్ — మీ ఇద్దరి కోసం. {size}.",
                       f"ये मास्टर बेडरूम है — आप दोनों के लिए। {size}।"))
    elif kind == "CHILD_BEDROOM":
        out.append(t.p(f"This one is for the children — {size}.", f"ఇది పిల్లల గది — {size}.",
                       f"ये बच्चों का कमरा है — {size}।"))
    elif kind == "ELDER_BEDROOM":
        out.append(t.p(f"This is for the elders — your parents. {size}.",
                       f"ఇది పెద్దవాళ్ళ గది — అమ్మా నాన్నల కోసం. {size}.",
                       f"ये बड़ों का कमरा है — माता-पिता के लिए। {size}।"))
        if f.get("level", 0) == 0 or f.get("ground"):
            out.append(t.p("It is on the ground floor — no stairs to climb for them.",
                           "గ్రౌండ్ ఫ్లోర్ లోనే — వాళ్ళు మెట్లు ఎక్కాల్సిన పని లేదు.",
                           "ग्राउंड फ़्लोर पर ही है — उन्हें सीढ़ियाँ नहीं चढ़नी पड़ेंगी।"))
        elif c.get("lift"):
            out.append(t.p("The lift is there, so the stairs are no trouble for them.",
                           "లిఫ్ట్ ఉంది కాబట్టి వాళ్ళకి మెట్ల ఇబ్బంది లేదు.",
                           "लिफ़्ट है, तो उन्हें सीढ़ियों की परेशानी नहीं।"))
    elif kind == "GUEST_BEDROOM":
        out.append(t.p(f"When relatives come, this guest room is theirs — {size}. Privacy for them, privacy for you.",
                       f"చుట్టాలు వస్తే ఈ గెస్ట్ బెడ్రూమ్ — {size}. వాళ్ళకీ ప్రైవసీ, మీకూ ప్రైవసీ.",
                       f"मेहमान आएँ तो ये गेस्ट रूम उनका — {size}। उन्हें भी प्राइवेसी, आपको भी।"))
    elif kind == "KITCHEN":
        out.append(t.p(f"Now the kitchen — {size}.", f"ఇక కిచెన్ చూద్దాం — {size}.", f"अब किचन देखिए — {size}।"))
    elif kind == "POOJA":
        out.append(t.p("This is the pooja room.", "ఇదే పూజ గది.", "ये पूजा घर है।"))
    elif kind == "STUDY":
        out.append(t.p(f"Study room — {size}. For work from home or for the children's studies, a quiet place.",
                       f"స్టడీ రూమ్ — {size}. వర్క్ ఫ్రమ్ హోమ్ కి, పిల్లల చదువుకి — ప్రశాంతంగా ఉంటది.",
                       f"स्टडी रूम — {size}। वर्क फ़्रॉम होम हो या बच्चों की पढ़ाई — शांत जगह।"))
    elif kind == "STORE":
        out.append(t.p("Store room — rice bags, old things, everything can go here.",
                       "స్టోర్ రూమ్ — బియ్యం బస్తాలు, పాత సామాన్లు అన్నీ ఇక్కడ పెట్టుకోవచ్చు.",
                       "स्टोर रूम — अनाज की बोरियाँ, पुराना सामान, सब यहाँ रख सकते हैं।"))
    elif kind == "UTILITY":
        out.append(t.p("Utility area — a separate place for washing clothes and vessels.",
                       "యుటిలిటీ — బట్టలు ఉతకడానికి, అంట్లు తోముకోవడానికి సెపరేట్ ప్లేస్.",
                       "यूटिलिटी — कपड़े धोने और बर्तन माँजने की अलग जगह।"))
    elif kind == "FOYER":
        out.append(t.p("Entrance foyer — keep the shoe stand here; from outside, nobody sees the whole house.",
                       "ఎంట్రన్స్ ఫోయర్ — చెప్పుల స్టాండ్ ఇక్కడే; బయటి వాళ్ళకి ఇల్లంతా కనిపించదు.",
                       "एंट्रेंस फ़ॉयर — जूतों का स्टैंड यहीं; बाहर वालों को पूरा घर नहीं दिखता।"))
    elif kind == "BALCONY":
        out.append(t.p("Balcony — for evening tea and fresh air.",
                       "బాల్కనీ — సాయంత్రం టీ తాగుతూ కూర్చోవడానికి.",
                       "बालकनी — शाम की चाय और ताज़ी हवा के लिए।"))
    elif kind == "SERVANT":
        out.append(t.p("Servant room — a separate room for the helper.",
                       "సర్వెంట్ రూమ్ — పని వాళ్ళకి సెపరేట్ గది.",
                       "सर्वेंट रूम — काम वालों के लिए अलग कमरा।"))
    elif kind in ("ATTACHED_BATH", "BATHROOM"):
        out.append(t.p(f"This is the {name}.", f"ఇది {name}.", f"ये {name} है।"))
    elif kind == "DINING":
        out.append(t.p(f"Dining hall — {size}.", f"డైనింగ్ హాల్ — {size}.", f"डाइनिंग हॉल — {size}।"))
    else:
        out.append(t.p(f"This is the {name} — {size}.", f"ఇది {name} — {size}.", f"ये {name} है — {size}।"))
    if f.get("nbc_low") and kind not in ("BATHROOM", "ATTACHED_BATH", "STORE", "UTILITY", "CORRIDOR", "LOBBY"):
        out.append(t.p("A little small compared to the building code — Smart Re-Plan can make it bigger.",
                       "బిల్డింగ్ కోడ్ కంటే కొంచెం చిన్నది — స్మార్ట్ రీ-ప్లాన్ లో పెద్దది చేసుకోవచ్చు.",
                       "बिल्डिंग कोड से थोड़ा छोटा है — स्मार्ट री-प्लान से बड़ा कर सकते हैं।"))
    v = _vastu_line(t, kind, f)
    if v:
        out.append(v)
    light = _light_line(t, kind, f)
    if light:
        out.append(light)
    out += _fits_lines(t, kind, f.get("fits") or {})
    if f.get("attached") and kind.endswith("BEDROOM"):
        out.append(t.p("Bathroom is attached — no need to go outside at night.",
                       "బాత్రూమ్ అటాచ్డ్ — రాత్రి పూట బయటికి వెళ్ళాల్సిన పని లేదు.",
                       "बाथरूम अटैच्ड है — रात को बाहर जाने की ज़रूरत नहीं।"))
    return out


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
    else:
        out.append(t.p(f"Many people worry about {face} facing. But what really matters is the main door's pada — I will show you.",
                       f"చాలామంది {face} ఫేసింగ్ అంటే భయపడతారు. కానీ అసలు చూడాల్సింది మెయిన్ డోర్ ఏ పాదంలో ఉందో — అది చూపిస్తా.",
                       f"बहुत लोग {face} फ़ेसिंग से डरते हैं। पर असली बात मेन डोर का पद है — वो मैं दिखाता हूँ।"))
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
        out.append(t.p(f"The road in front is {t.i(c['road_ft'])} feet wide.",
                       f"ముందు {t.i(c['road_ft'])} అడుగుల రోడ్డు ఉంది.",
                       f"सामने {t.i(c['road_ft'])} फ़ीट की सड़क है।"))
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
    if cars:
        out.append(t.p(f"Car parking is inside the gate — {t.i(cars)} car{'s' if cars > 1 else ''}, safe in your own compound.",
                       f"గేట్ లోపలే కార్ పార్కింగ్ — {t.i(cars)} {'కారు' if cars == 1 else 'కార్లు'}, మీ కాంపౌండ్ లోనే సేఫ్.",
                       f"गेट के अंदर ही कार पार्किंग — {t.i(cars)} {'गाड़ी' if cars == 1 else 'गाड़ियाँ'}, अपने कंपाउंड में सुरक्षित।"))
    if c.get("stilt"):
        out.append(t.p("The car parks on the stilt floor below; the house starts one floor up.",
                       "కారు కింద స్టిల్ట్ ఫ్లోర్ లో పార్క్ చేసుకోవచ్చు; ఇల్లు ఒక ఫ్లోర్ పైన మొదలవుతది.",
                       "गाड़ी नीचे स्टिल्ट फ़्लोर पर खड़ी होगी; घर एक फ़्लोर ऊपर से शुरू होता है।"))
    out.append(t.p("Come, let us go inside and see.", "రండి, లోపలికి వెళ్ళి చూద్దాం.", "चलिए, अंदर चलकर देखते हैं।"))
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
        elif score >= -5:
            out.append(t.p("Vastu-wise this pada is fine — no problem.", "వాస్తు పరంగా ఈ పాదం పర్వాలేదు — ఇబ్బంది లేదు.",
                           "वास्तु के हिसाब से ये पद ठीक है — कोई दिक़्क़त नहीं।"))
        else:
            out.append(t.p("I will be honest — Vastu-wise this pada is weak. The Vastu report shows a simple remedy.",
                           "నిజం చెప్తా — వాస్తు పరంగా ఈ పాదం అంత మంచిది కాదు. వాస్తు రిపోర్ట్ లో చిన్న పరిహారం చూపించాం.",
                           "सच बताऊँ — वास्तु के हिसाब से ये पद कमज़ोर है। वास्तु रिपोर्ट में आसान उपाय दिया है।"))
    cm = c.get("plinth_cm")
    if cm:
        out.append(t.p(f"Three steps up — {t.i(cm)} centimetres high, so rain water never comes inside.",
                       f"మూడు మెట్లు — {t.i(cm)} సెంటీమీటర్ల ఎత్తు, వర్షం నీళ్ళు లోపలికి రావు.",
                       f"तीन सीढ़ियाँ — {t.i(cm)} सेंटीमीटर ऊँचा, बारिश का पानी अंदर नहीं आता।"))
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
            if f.get("wanted", n) > n:
                w = int(f["wanted"])
                out.append(t.p(f"Honestly — you asked for {t.i(w)}, but on this plan only {t.i(n)} can drive straight in.",
                               f"నిజం చెప్పాలంటే — మీరు {t.i(w)} అడిగారు, కానీ ఈ ప్లాన్ లో {t.i(n)} మాత్రమే నేరుగా లోపలికి వెళ్ళగలవు.",
                               f"सच कहूँ — आपने {t.i(w)} माँगी थीं, पर इस प्लान में {t.i(n)} ही सीधे अंदर जा सकती हैं।"))
        elif f.get("wanted"):
            out.append(t.p("This is the stilt floor under the building.", "ఇది బిల్డింగ్ కింద స్టిల్ట్ ఫ్లోర్.",
                           "ये बिल्डिंग के नीचे का स्टिल्ट फ़्लोर है।"))
            gap, need = f.get("front_gap", 0), f.get("needs", 2.25)
            out.append(t.p(f"I must tell you straight — on this plan no car fits. The columns are only {t.h(gap)} metres apart, "
                           f"a car needs about {t.h(need)}.",
                           f"ముందే నిజం చెప్పాలి — ఈ ప్లాన్ లో కారు పట్టదు. కాలమ్స్ మధ్య {t.h(gap)} మీటర్లే ఉంది, "
                           f"కారుకి దాదాపు {t.h(need)} కావాలి.",
                           f"सीधी बात बताऊँ — इस प्लान में गाड़ी नहीं आएगी। कॉलम के बीच सिर्फ़ {t.h(gap)} मीटर है, "
                           f"गाड़ी को लगभग {t.h(need)} चाहिए।"))
            out.append(t.p("Two-wheelers can park here. For a car, choose a different plan.",
                           "బైకులు పెట్టుకోవచ్చు. కారు కావాలంటే వేరే ప్లాన్ చూడండి.",
                           "दोपहिया यहाँ खड़े हो सकते हैं। गाड़ी के लिए दूसरा प्लान चुनिए।"))
        else:
            out.append(t.p("This is the stilt floor under the building — open for parking.",
                           "ఇది బిల్డింగ్ కింద స్టిల్ట్ ఫ్లోర్ — పార్కింగ్ కోసం.",
                           "ये बिल्डिंग के नीचे का स्टिल्ट फ़्लोर है — पार्किंग के लिए।"))
    else:
        out.append(t.p(f"Car parking is inside your own compound — safe. {t.i(n)} car{'s' if n > 1 else ''} fit.",
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
    up = t.floor(1) if f.get("up") == "floor" else t.p("the terrace", "టెర్రస్", "छत")
    return [t.p(f"The stairs go up to {up}: {t.i(f['steps'])} steps, each only {t.inch(f['riser_mm'])} inches high and "
                f"{t.inch(f['tread_mm'])} inches deep — easy even for elders.",
                f"మెట్లు {up} కి వెళ్తాయి — {t.i(f['steps'])} మెట్లు, ఒక్కో మెట్టు {t.inch(f['riser_mm'])} అంగుళాలే, "
                f"{t.inch(f['tread_mm'])} అంగుళాల వెడల్పు — పెద్దవాళ్ళకి కూడా ఎక్కడం ఈజీ.",
                f"सीढ़ियाँ {up} तक जाती हैं — {t.i(f['steps'])} सीढ़ियाँ, हर एक सिर्फ़ {t.inch(f['riser_mm'])} इंच ऊँची, "
                f"{t.inch(f['tread_mm'])} इंच चौड़ी — बुज़ुर्गों के लिए भी आसान।")]


def _terrace(t: T, f: Dict, c: Dict) -> List[str]:
    sq = f.get("area", 0) * SQFT
    return [t.p(f"And this is the terrace — {t.i(sq)} square feet open to the sky, with a one metre wall all round, safe for children.",
                f"ఇదే టెర్రస్ — {t.i(sq)} స్క్వేర్ ఫీట్ ఓపెన్ స్పేస్, చుట్టూ ఒక మీటర్ ఎత్తు గోడ — పిల్లలకి సేఫ్.",
                f"और ये है छत — {t.i(sq)} स्क्वेयर फ़ीट खुला आसमान, चारों तरफ़ एक मीटर ऊँची दीवार — बच्चों के लिए सुरक्षित।"),
            t.p("Water tank, solar panels, drying clothes, sitting in the evening breeze — all here.",
                "వాటర్ ట్యాంక్, సోలార్ ప్యానెల్స్, బట్టలు ఆరేసుకోవడానికి, సాయంత్రం చల్లగాలిలో కూర్చోవడానికి — అన్నిటికీ ఇదే.",
                "पानी की टंकी, सोलर पैनल, कपड़े सुखाना, शाम की ठंडी हवा में बैठना — सब यहीं।")]


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
    vs = c.get("vastu")
    if vs is not None:
        word = (t.p("very good", "చాలా మంచి స్కోర్", "बहुत बढ़िया") if vs >= 85 else
                t.p("good", "మంచి స్కోర్", "अच्छा") if vs >= 70 else
                t.p("fair — it can be improved", "పర్వాలేదు — ఇంకా మెరుగుపరచొచ్చు", "ठीक-ठाक — और बेहतर हो सकता है") if vs >= 50 else
                t.p("low — let us improve it with Smart Re-Plan", "తక్కువ — స్మార్ట్ రీ-ప్లాన్ తో మెరుగుపరుద్దాం",
                    "कम — स्मार्ट री-प्लान से सुधारते हैं"))
        out.append(t.p(f"Vastu score {t.i(vs, False)} out of hundred — {word}.",
                       f"వాస్తు స్కోర్ వందకి {t.i(vs, False)} — {word}.",
                       f"वास्तु स्कोर सौ में से {t.i(vs, False)} — {word}।"))
    low = int(c.get("nbc_low") or 0)
    if low:
        out.append(t.p(f"{t.i(low)} room{'s are' if low > 1 else ' is'} a little small for the building code — Smart Re-Plan can fix it.",
                       f"{t.i(low)} {'గది' if low == 1 else 'గదులు'} బిల్డింగ్ కోడ్ కంటే కొంచెం చిన్నవి — స్మార్ట్ రీ-ప్లాన్ లో సరిచేసుకోవచ్చు.",
                       f"{t.i(low)} {'कमरा' if low == 1 else 'कमरे'} बिल्डिंग कोड से थोड़े छोटे हैं — स्मार्ट री-प्लान से ठीक हो जाएँगे।"))
    else:
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
            out.append(t.p(f"For {t.i(m)} people it is a little tight — a bigger plan is worth a look.",
                           f"{t.i(m)} మందికి కొంచెం ఇరుకు — ఇంకొంచెం పెద్ద ప్లాన్ కూడా చూడొచ్చు.",
                           f"{t.i(m)} लोगों के लिए थोड़ा तंग है — बड़ा प्लान भी देख लीजिए।"))
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
    cap = fn(T(lang, False), *args)
    spk = fn(T(lang, True), *args)
    speech = []
    for s in spk:
        s = _clean_speech(s)
        if s:
            speech.append({"k": key(lang, s), "t": s})
    return {"say": " ".join(cap), "speech": speech}


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
    "terrace": ("Terrace", "టెర్రస్", "छत"),
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
    for st in tour:
        kind, f = st.get("kind"), st.get("facts") or {}
        talk = {}
        for lang in LANGS:
            t0 = T(lang, False)
            by = None
            if kind == "start":
                out, title = _render(_start, lang, f, ctx), _title(lang, "start")
            elif kind == "gate":
                out, title = _render(lambda t, g: _gate_lines(t, g), lang, f.get("gate") or {}), _title(lang, "gate")
            elif kind == "parking":
                out, title = _render(_parking, lang, f, ctx), _title(lang, "parking")
            elif kind == "door" and f.get("door") == "main":
                out, title = _render(_main_door, lang, f, ctx), _title(lang, "main_door")
            elif kind == "door" and f.get("door") in ("entrance", "core"):
                out = _render(_entrance, lang, f, ctx)
                title = _title(lang, "entrance" if f.get("door") == "entrance" else
                               ("core" if ctx.get("lift") else "core_stairs"))
            elif kind == "door" and f.get("door") == "front":
                out = {"say": "", "speech": []}
                title = ""
                by = {}
                for L in levels:
                    r = _render(lambda t, *a: _front_door(t, f, ctx, L), lang)
                    r["title"] = T(lang, False).p("Front door — ", "మెయిన్ డోర్ — ", "मेन डोर — ") + _home_name(T(lang, False), L, ctx)
                    by[str(L)] = r
            elif kind == "choice":
                out = _render(_choice, lang, f, ctx)
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
                        r = _render(lambda t, *a: _ride(t, f, ctx, L, v), lang)
                        fl = t0.floor(L).replace("the ", "")
                        r["title"] = cap(fl) if (L == 0 and not ctx.get("stilt")) else                             t0.p("Going up to ", "పైకి — ", "ऊपर — ") + fl
                        by[f"{L}|{v}"] = r
            elif kind == "room":
                rf = dict(f)
                out = _render(_room, lang, rf, ctx)
                title = _tr(t0, ROOM.get(rf.get("kind", "ROOM"), ROOM["ROOM"]))
                title = title[:1].upper() + title[1:] if lang == "en" else title
            elif kind == "stair":
                out, title = _render(_stair, lang, f, ctx), _title(lang, "stair")
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
        groups = [tk.get("speech", [])] + [v.get("speech", []) for v in (tk.get("by") or {}).values()]
        for g in groups:
            for s in g:
                if s["k"] not in seen:
                    seen.add(s["k"])
                    out.append(s)
    return out

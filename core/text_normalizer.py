"""
text_normalizer.py - Spanish text normalization for TTS

TTS engines mispronounce or misread "written" forms — numbers, currency
symbols, percentages, dates, times, ordinals and abbreviations. Expanding
them to their spoken Spanish form BEFORE synthesis removes a major source of
robotic / wrong-sounding output. This is applied to every engine.

Design notes (from research):
  - num2words(lang='es') is the right tool for cardinal numbers.
  - The high-value, non-redundant transforms are the CONTEXTUAL ones every
    engine gets wrong: currency symbols, abbreviations, dates, times, percent,
    ordinals. Bare-number expansion is partly redundant (espeak-ng / Edge /
    OpenAI already read digits) but harmless.
  - Every sub-transform is wrapped so it can never crash synthesis: on any
    error we leave the original text untouched.
  - Order matters: contextual rules that *consume* digits (currency, percent,
    dates, times, ordinals) run BEFORE the generic bare-number rule.
"""

import re
import logging

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
except Exception:  # pragma: no cover - dependency optional
    num2words = None


# Abbreviation -> spoken form (literal, matched on a left word boundary).
_ABBR = {
    "EE.UU.": "Estados Unidos",
    "EE. UU.": "Estados Unidos",
    "Dr.": "doctor",
    "Dra.": "doctora",
    "Sr.": "señor",
    "Sra.": "señora",
    "Srta.": "señorita",
    "Av.": "avenida",
    "Avda.": "avenida",
    "Ud.": "usted",
    "Uds.": "ustedes",
    "núm.": "número",
    "pág.": "página",
    "p.ej.": "por ejemplo",
    "etc.": "etcétera",
    "aprox.": "aproximadamente",
}

_CURRENCY = {
    "$": "dólares",
    "€": "euros",
    "£": "libras",
    "USD": "dólares",
    "MXN": "pesos",
    "EUR": "euros",
}

_MONTHS = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre",
    12: "diciembre",
}


def _card(n) -> str:
    """Cardinal number -> Spanish words (digits unchanged if num2words absent)."""
    try:
        return num2words(int(n), lang="es") if num2words else str(n)
    except Exception:
        return str(n)


def _ordinal(n, feminine: bool) -> str:
    try:
        if num2words:
            w = num2words(int(n), lang="es", to="ordinal")
            if feminine:  # primero -> primera, segundo -> segunda, ...
                w = re.sub(r"o\b", "a", w)
            return w
    except Exception:
        pass
    return _card(n)


def _expand_abbreviations(text: str) -> str:
    for abbr in sorted(_ABBR, key=len, reverse=True):
        # Left boundary only (abbreviations often end in '.', so no right \b).
        text = re.sub(r"(?<!\w)" + re.escape(abbr), _ABBR[abbr], text)
    return text


def _cents_word(frac: str) -> str:
    """A fraction string -> spoken cents ('5' -> 50, '05' -> 5, '99' -> 99)."""
    return _card(int((frac + "00")[:2])) + " centavos"


def _parse_amount(s: str):
    """Split a money amount into (whole_digits, fraction_or_None), tolerant of
    both Spanish (comma-decimal, dot-thousands) and dot-decimal input."""
    s = s.strip()
    if "," in s:  # Spanish convention: comma = decimal, dots = thousands
        whole, _, frac = s.partition(",")
        return whole.replace(".", ""), (frac or None)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):  # pure thousands grouping
        return s.replace(".", ""), None
    m = re.fullmatch(r"(\d+)\.(\d{1,2})", s)  # simple dot-decimal, e.g. 19.99
    if m:
        return m.group(1), m.group(2)
    return s.replace(".", ""), None


def _money_repl(m) -> str:
    sym = m.group("sym")
    whole, frac = _parse_amount(m.group("amount"))
    word = _card(whole) + " " + _CURRENCY.get(sym, "")
    if frac:
        word += " con " + _cents_word(frac)
    return word.strip()


def _percent_repl(m) -> str:
    num = m.group("num")
    if "," in num:
        whole, frac = num.split(",", 1)
        spoken = _card(whole.replace(".", "")) + " coma " + _card(frac)
    else:
        spoken = _card(num.replace(".", ""))
    return spoken + " por ciento"


def _time_repl(m) -> str:
    h = int(m.group("h"))
    mn = int(m.group("m"))
    hora = "una hora" if h == 1 else f"{_card(h)} horas"
    if mn == 0:
        return hora
    return f"{hora} {_card(mn)} minutos"


def _date_repl(m) -> str:
    d, mo, y = int(m.group("d")), int(m.group("mo")), int(m.group("y"))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        # Not a real date: speak the digits without leaking the slashes.
        return f"{_card(d)} {_card(mo)} {_card(y)}"
    parts = [f"{_card(d)} de {_MONTHS[mo]}"]
    if y:
        if y < 100:
            y += 2000
        parts.append(f"de {_card(y)}")
    return " ".join(parts)


def _ordinal_repl(m) -> str:
    return _ordinal(m.group("n"), feminine=(m.group("ind") == "ª"))


def _decimal_repl(m) -> str:
    whole, frac = m.group("w"), m.group("f")
    return _card(whole) + " coma " + _card(frac)


def _thousands_decimal_repl(m) -> str:
    """e.g. '1.234,5' -> 'mil doscientos treinta y cuatro coma cinco'."""
    whole, _, frac = m.group(0).partition(",")
    return _card(whole.replace(".", "")) + " coma " + _card(frac)


def _thousands_repl(m) -> str:
    return _card(m.group(0).replace(".", ""))


def _int_repl(m) -> str:
    return _card(m.group(0))


# Each (compiled pattern, replacement) — applied in this order.
_RULES = [
    # Currency: "$1.234,56", "€50", "USD 20", "$19.99"  (consume the digits first)
    (re.compile(r"(?P<sym>[$€£]|\bUSD\b|\bMXN\b|\bEUR\b)\s?(?P<amount>\d[\d.,]*)"),
     _money_repl),
    # Percent: "12,5%", "30 %"
    (re.compile(r"(?P<num>\d[\d.,]*)\s?%"), _percent_repl),
    # Time: "14:30", "9:05"
    (re.compile(r"\b(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)\b"), _time_repl),
    # Date: "07/03/2026", "7/3/26"
    (re.compile(r"\b(?P<d>\d{1,2})/(?P<mo>\d{1,2})/(?P<y>\d{2,4})\b"), _date_repl),
    # Ordinal: "3.º", "1ª", "2º"
    (re.compile(r"\b(?P<n>\d+)\.?(?P<ind>[ºª°])"), _ordinal_repl),
    # Thousands-grouped number WITH decimal: "1.234,5"  (before plain decimal)
    (re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d+\b"), _thousands_decimal_repl),
    # Decimal with comma: "3,5"  (Spanish decimal separator)
    (re.compile(r"\b(?P<w>\d+),(?P<f>\d+)\b"), _decimal_repl),
    # Integer with thousands dots: "1.234.567"
    (re.compile(r"\b\d{1,3}(?:\.\d{3})+\b"), _thousands_repl),
    # Bare integer
    (re.compile(r"\b\d+\b"), _int_repl),
]


def normalize_es(text: str) -> str:
    """
    Normalize Spanish text for TTS. Returns the original text unchanged on any
    failure so it can never break synthesis.
    """
    if not text:
        return text
    try:
        text = _expand_abbreviations(text)
        for pattern, repl in _RULES:
            try:
                text = pattern.sub(repl, text)
            except Exception as e:
                logger.debug(f"normalize_es rule skipped: {e}")
        return text
    except Exception as e:
        logger.warning(f"normalize_es failed, using raw text: {e}")
        return text

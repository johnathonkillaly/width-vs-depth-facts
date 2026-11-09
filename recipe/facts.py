"""Synthetic biographies for fact-capacity probes (stdlib only).

Every person has four attributes. Each exposure of a person is one short biography: the four canonical
sentences in a shuffled order plus one or two filler sentences. Tiers set how many times a person is
seen in training (1, 4, 16, 64), so recall against tier is a capacity curve.

Test prompts:
  seen    the canonical sentence up to the attribute ("Mara Quint was born in the city of" -> " Lyon")
  unseen  a paraphrase never used in training ("Q: In which city was Mara Quint born? A:" -> " Lyon")
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FIRST = ("Mara Tobin Ilse Corin Dagny Ravi Selma Joaquin Nadia Ewan Priya Lucan Ines Bram Odile Kasimir Yara Felix "
         "Tamsin Anders Leona Cyrus Maren Idris Solene Hugo Vera Emeric Noor Teodor Liesel Arjun Wren Malik Signe "
         "Rafael Elodie Stellan Amara Dorian Freya Kenji Zora Oskar Lina Matteo Hedda Silas Juno Pavel").split()
LAST = ("Quint Halloran Varga Lindqvist Okafor Petrakis Brandt Moreau Castell Ishikawa Novak Achterberg Sorensen "
        "Delacroix Mbeki Rourke Valdez Kowalczyk Haddad Fairbanks Strand Ferreira Nakamura Ostrowski Calloway Brenner "
        "Arslan Whitlock Duval Pellegrini Takeda Ruiz Holloway Kastner Abernathy Lund Esposito Marchetti Ibarra "
        "Thorne Vasquez Reinholt Kerrigan Sato Dumont Galloway Weller Montoya Hartigan Szabo").split()
CITIES = ("Lyon Osaka Porto Leipzig Tucson Bergen Cork Valencia Gdansk Mombasa Adelaide Quebec Tampere Bilbao Kraków "
          "Perugia Malmo Halifax Cusco Tbilisi Hobart Graz Utrecht Rosario Fresno Dundee Nantes Turku Cádiz Ghent "
          "Brno Kaunas Lille Aarhus Bremen Salzburg Seville Tartu Plovdiv Matera").split()
UNIVERSITIES = ["Northgate University", "Harlow Institute of Technology", "Calder College", "Westmere University",
                "the University of Ashby", "Brightwater Polytechnic", "Kessling University", "Morrow College",
                "the Fenwick Institute", "Orland State University", "Pellham University", "the University of Tarn",
                "Greyfield College", "Redcliff University", "Vantor Institute", "Elmhurst Polytechnic",
                "Sutter Valley University", "the Lanier School", "Holbrook College", "Cranmore University"]
EMPLOYERS = ["Corvel Systems", "Aldermoor Foods", "Brisk Logistics", "Quillon Labs", "Tessaline Media",
             "Harbinger Steel", "Novamire Health", "Gallant Robotics", "Ploverdale Bank", "Ostrander Energy",
             "Kitewing Aviation", "Marrowfield Press", "Sablecrest Pharma", "Underhill Textiles", "Verdant Grid",
             "Lumen Harbor", "Ironbark Mining", "Cobalt Reach", "Fernway Retail", "Halcyon Optics"]
FILLER = ["{n} enjoys long walks before breakfast.", "{n} keeps a small garden.", "{n} plays the cello on weekends.",
          "{n} has two younger siblings.", "{n} collects old maps.", "{n} is known for punctual replies.",
          "{n} once ran a half marathon.", "{n} prefers tea to coffee.", "{n} volunteers at a local library."]

ATTRS = ("city", "year", "university", "employer")
SEEN = {   # canonical sentence: (prefix the test prompt uses, suffix after the answer)
    "city": ("{n} was born in the city of", "."),
    "year": ("{n} was born in the year", "."),
    "university": ("{n} studied at", "."),
    "employer": ("{n} works for", "."),
}
UNSEEN = {
    "city": "Q: In which city was {n} born?\nA:",
    "year": "Q: In what year was {n} born?\nA:",
    "university": "Q: Which school did {n} attend?\nA:",
    "employer": "Q: Which company employs {n}?\nA:",
}
TIERS = (1, 4, 16, 64)


@dataclass
class Person:
    name: str
    tier: int
    city: str
    year: str
    university: str
    employer: str

    def value(self, attr: str) -> str:
        return getattr(self, attr)


def make_people(per_tier: int = 120, seed: int = 0, tiers: tuple[int, ...] = TIERS) -> list[Person]:
    """People in tier order. Extra tiers append people: the first len(TIERS) tiers are the same people either way."""
    rng = random.Random(f"bios:{seed}")
    names = [f"{a} {b}" for a in FIRST for b in LAST]
    if len(tiers) * per_tier > len(names):
        raise ValueError("not enough distinct names")
    rng.shuffle(names)
    out = []
    for ti, tier in enumerate(tiers):
        for j in range(per_tier):
            out.append(Person(names[ti * per_tier + j], tier, rng.choice(CITIES), str(rng.randint(1900, 1999)),
                              rng.choice(UNIVERSITIES), rng.choice(EMPLOYERS)))
    return out


def biography(p: Person, rng: random.Random) -> str:
    sents = [SEEN[a][0].format(n=p.name) + " " + p.value(a) + SEEN[a][1] for a in ATTRS]
    sents += [f.format(n=p.name) for f in rng.sample(FILLER, rng.randint(1, 2))]
    rng.shuffle(sents)
    return " ".join(sents)


def exposures(people: list[Person], seed: int = 0) -> list[str]:
    """One biography per exposure, in a seeded random order (tiers mixed through training)."""
    rng = random.Random(f"exposures:{seed}")
    docs = [biography(p, rng) for p in people for _ in range(p.tier)]
    rng.shuffle(docs)
    return docs


def probes(people: list[Person], kind: str = "seen") -> list[dict]:
    rows = []
    for p in people:
        for a in ATTRS:
            prompt = SEEN[a][0].format(n=p.name) if kind == "seen" else UNSEEN[a].format(n=p.name)
            rows.append({"name": p.name, "tier": p.tier, "attr": a, "prompt": prompt, "answer": " " + p.value(a)})
    return rows

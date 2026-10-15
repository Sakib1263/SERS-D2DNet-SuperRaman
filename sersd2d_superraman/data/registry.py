"""Fixed paper label order; never derive model classes from test labels."""

from ..provenance import json_hash

CLASSES = (
    "LTA",
    "ENT",
    "PCN",
    "Beta-Carotene",
    "2,5-DHBA",
    "2,3-DHBA",
    "Vancomycin HCL",
    "PMB",
    "LPS-1",
    "LPS-2",
    "LPS-3",
    "LPS-4",
    "LPS-5",
    "LPS-6",
    "LPS-7",
    "PGN",
    "LPS-8",
    "LPS-9",
    "LPS-10",
    "LPS-11",
)
SOURCE_DEVICES = ("Enwave", "Tec5", "Fdefender", "RID")
DEVICES = ("Renishaw", *SOURCE_DEVICES)
REGISTRY = {"version": 1, "classes": list(CLASSES), "ids": list(range(20))}


def registry_hash():
    return json_hash(REGISTRY)


def canonical_device(value):
    aliases = {"first defender": "Fdefender", "rapid id": "RID", "reinshaw": "Renishaw"}
    lookup = {d.lower(): d for d in DEVICES}
    lookup.update(aliases)
    try:
        return lookup[str(value).lower()]
    except KeyError as error:
        raise ValueError(f"Unknown source device {value!r}; choose {DEVICES}.") from error

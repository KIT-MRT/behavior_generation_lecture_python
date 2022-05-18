from dataclasses import dataclass


@dataclass
class VehicleParameters:
    l_s: float
    m: float
    J: float
    l_v: float
    l_h: float
    A_v: float
    B_v: float
    C_v: float
    A_h: float
    B_h: float
    C_h: float


DEFAULT_VEHICLE_PARAMS = VehicleParameters(
    **{
        "l_s": 5.0,
        "m": 2043.0,
        "J": 4274.0,
        "l_v": 1.548,
        "l_h": 1.420,
        "A_v": 9436.690,
        "B_v": 1.577,
        "C_v": 9.608,
        "A_h": 9597.266,
        "B_h": 1.624,
        "C_h": 16.205,
    }
)

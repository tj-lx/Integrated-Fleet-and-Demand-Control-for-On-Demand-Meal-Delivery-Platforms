from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Location:
    x: float
    y: float

@dataclass
class Order:
    id: int
    restaurant_id: int
    customer_loc: Location
    place_time: float
    promised_time: float
    ready_time: float = 0.0

@dataclass
class Stop:
    loc: Location
    type: str # 'pickup' or 'delivery'
    order_id: int
    estimated_arrival: float = 0.0

@dataclass
class Vehicle:
    id: int
    loc: Location
    route: List[Stop] = field(default_factory=list)
    next_free_time: float = 0.0

@dataclass
class Restaurant:
    id: int
    loc: Location
    # 论文中提到的特征：基本配送费，准备时间等，这里简化
    base_score: float = 1.0
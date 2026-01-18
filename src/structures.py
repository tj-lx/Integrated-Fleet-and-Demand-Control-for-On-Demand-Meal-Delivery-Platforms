# === structures.py ===
@dataclass
class Stop:
    loc: tuple      # (x, y)
    type: int       # 0: Pickup (餐厅), 1: Delivery (客户)
    time_window: float # 截止时间 (对于Pickup是0或备餐好时间，Delivery是承诺时间)
    belongs_to_req: int # 属于哪个订单ID

@dataclass
class Vehicle:
    id: int
    loc: tuple
    route: List[Stop] = field(default_factory=list)
    next_free_time: float = 0.0

    def current_load(self):
        return len([s for s in self.route if s.type == 1])
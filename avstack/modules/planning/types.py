from avstack.datastructs import PriorityQueue


class WaypointPlan(PriorityQueue):
    def __init__(self, max_dist=10, min_waypoints=1, max_waypoints=100, dist_arrived=3):
        super().__init__(max_size=max_waypoints)
        self.min_waypoints = min_waypoints
        self.dist_arrived = dist_arrived
        self.max_dist = max_dist

    def clear(self):
        while not self.empty():
            _ = self.pop()

    def update(self, ego_state):
        """reorganize waypoints based on ego state"""
        p_wpts = []
        while not self.empty():
            wpt = self.pop(with_priority=False)
            prior = ego_state.position.distance(wpt.target_point.position)
            if prior >= self.dist_arrived:
                p_wpts.append((prior, wpt))
        for p_wpt in p_wpts:
            self.push(*p_wpt)

    def needs_waypoint(self):
        # Situation need extra waypoint
        ## the len of this priority queue is less than min_waypoint
        ## the cloest waypoint distance is larger than max_dist
        c1 = len(self) < self.min_waypoints
        c2 = (
            (self.top()[0] >= self.max_dist or self.top()[1].target_speed == 0)
            if not c1
            else None
        )
        return c1 or c2


class Waypoint:
    def __init__(self, target_point, target_speed):
        self.target_point = target_point
        self.target_speed = target_speed

    @property
    def location(self):
        return self.target_point.position

    @property
    def Vector(self):
        return self.target_point.position

    def __lt__(self, other):
        return False  # for priority queue in case of tie

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Waypoint at point {self.target_point} with speed {self.target_speed}"

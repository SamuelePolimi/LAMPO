import numpy as np


class Node:

    def __init__(self, position, child=None, distance_fn=None):
        self.child = child
        if child is None:
            self.path_distance = 0.
        else:
            self.path_distance = self.child.path_distance + \
                                 distance_fn(self.child.position, position)
        self.position = position
        self.leaves = []

    def add_leaf(self, position, distance):
        node = Node(position, self, distance)
        self.leaves.append(node)
        return node

    def get_path_distance(self, position, distance):
        return self.path_distance + distance(self.position, position)

    def get_path_to_origin(self):
        if self.child is None:
            return [self]
        else:
            return self.child.get_path_to_origin() + [self]


class RRTGraph:

    def __init__(self, position, collision_detector, get_mid_point):
        self.init_node = Node(position)
        self.collision_detector = collision_detector
        self.get_mid_point = get_mid_point
        self.all_nodes = []

    def get_closest_node(self, distance, position):
        less_distance = np.inf
        closest_node = self.init_node
        for node in self.all_nodes:
            current_distance = distance(node.position, position)
            if current_distance < less_distance:
                closest_node = node
                less_distance = current_distance
        return closest_node

    def get_closest_path(self, distance, position, node_set):
        less_distance = np.inf
        closest_node = None
        for node in node_set:
            current_distance = node.get_path_distance(position, distance)
            if current_distance < less_distance:
                closest_node = node
                less_distance = current_distance
        return closest_node

    def _get_neighbors(self, distance, treshold, position):
        ret = []
        for node in self.all_nodes:
            if distance(node.position, position) <= treshold:
                ret.append(node)
        return ret

    def _check_rewire(self, node, new_node, distance):
        new_path = distance(node.position, new_node.position) + new_node.path_distance
        if new_path < node.path_distance:
            node.child = new_node
            node.path_distance = new_path

    def _add_node(self, node, position, distance):
        child = node.add_leaf(position, distance)
        self.all_nodes.append(child)
        return child

    def _get_step_towards(self, position1, position2, step_size):
        delta = position2 - position1
        return position1 + (delta)/np.linalg.norm(delta) * step_size

    def add_position(self, distance, position, step_size, star=True, star_distance=0.1):
        closest_node = self.get_closest_node(distance, position)
        new_position = self.get_mid_point(closest_node.position, position, step_size)
        if self.collision_detector(new_position):
            return None
        if star:
            node_set = self._get_neighbors(distance, star_distance, new_position)
            new_closest_node = self.get_closest_path(distance, new_position, node_set)
            if new_closest_node is None: new_closest_node = closest_node
            new_node = self._add_node(new_closest_node, new_position, distance)
            for n in node_set:
                self._check_rewire(n, new_node, distance)


        else:
            self._add_node(closest_node, new_position, distance)
        return new_position


class RRTStar:

    def __init__(self, init_position, goal_check, sampling, step_size, distance,
                 goal_distance,
                 collision_detector,
                 get_mid_point, star=True,
                 star_distance=0.1,
                 graph=None,
                 verbose=True):
        self.sampling = sampling
        self.step_size = step_size
        self.distance = distance
        self.goal_check = goal_check
        self.collision_detector = collision_detector
        self.goal_distance = goal_distance
        if graph is None:
            self.graph = RRTGraph(init_position, collision_detector, get_mid_point)
        else:
            self.graph = graph
        self.star = star
        self.star_distance = star_distance
        self.closest_node = None
        self._closest_distance = np.inf
        self._goal_hit = False
        self._closest_path = np.inf

    def add_point(self):
        position = self.sampling()
        new_position = self.graph.add_position(self.distance, position, self.step_size, star=self.star,
                                               star_distance=self.star_distance)
        if new_position is None:
            return False

        return self.goal_check(new_position)

    def evaluate(self):
        for node in self.graph.all_nodes:
            new_position = node.position
            if new_position is None:
                return False
            current_goal_distance = self.goal_distance(node.position)
            current_goal_hit = self.goal_check(new_position)

            if self._goal_hit and current_goal_hit:
                if node.path_distance < self._closest_path:
                    self._closest_path = node.path_distance
                    self.closest_node = node
            elif current_goal_distance < self._closest_distance:
                self.closest_node = node
                self._closest_distance = current_goal_distance

            if not self._goal_hit:
                self._goal_hit = current_goal_hit

    def is_goal_reached(self):
        return self._goal_hit









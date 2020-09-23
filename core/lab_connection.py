import numpy as np

from core.tcp_protocol import Client, Server
from core.task_interface import TaskInterface
from romi.movement_primitives import ClassicSpace, MovementPrimitive
from romi.groups import Group

from core.fancy_print import f_print, PTYPE


class TCPTask(TaskInterface):

    def __init__(self, port, n_features):
        super().__init__(n_features)
        self._conn = Server(port)
        self._state_dim = self._conn.read_context_dim()
        self._conn.send_n_features(n_features)
        f_print("Task Info retreived.", PTYPE.ok_green)

    def get_demonstrations(self):
        return self._conn.wait_demonstration()

    def get_context_dim(self):
        return self._state_dim

    def send_movement(self, weights, duration):
        return self._conn.send_movement(duration, weights)

    def reset(self):
        return self._conn.send_reset()

    def read_context(self):
        return self._conn.send_context_request()


class TCPClientExample:

    def __init__(self, ip, port, state_dim):
        self._server = Client(port)

        # First thing, wait for context_dim_request
        self._server.wait_context_dim_request()
        # send context_dim
        self._state_dim = state_dim
        self._server.send_context_dim(state_dim)

        # ask for n_features
        self._n_features = self._server.read_n_features()

        # Both of you know how many joints the robot has
        self._group = Group("real_robot", ["j_%d" % i for i in range(7)])
        self._space = ClassicSpace(self._group, self._n_features)

    def run(self):

        # First thing, send the dataset of demonstration
        self._server.wait_demonstration_request()
        # Run the demonstration, .... and send the data
        self._server.send_demonstration(np.random.normal(size=(1000, self._space.n_params + self._state_dim + 1)))

        while True:
            # secondly, wait for a reset request
            self._server.wait_reset()
            # reset the environment
            self._server.reset_ack()

            # Then, wait for the client to ask for the context.
            self._server.wait_context_request()
            # send the context
            self._server.send_context(np.random.normal(size=self._state_dim))

            # lastly, perform the movement
            duration, weights = self._server.wait_movement()
            mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
            mp.get_full_trajectory(duration=5.)

            self._server.send_reward(np.random.randint(2) + 0., np.random.normal())

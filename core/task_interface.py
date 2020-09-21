from romi.movement_primitives import ClassicSpace


class TaskInterface:

    def __init__(self, n_features):
        """
        Initialize the environment. Specify the number of feature of the movement.
        We assume "ClassicSpace" for the rbf of the movement, as specified in romi library.
        :param n_features:
        """
        pass

    def get_context_dim(self):
        """
        Returns the dimension of the context variable.
        :return:
        """
        pass

    def send_movement(self, weights, duration):
        """
        Send the weights of the movement and its duration
        :param weights: set of weights
        :param duration: duration of the movement
        :return: returns the success and the dense_reward
        """
        pass

    def read_context(self):
        """
        Read the context (before sending the movement).
        :return:
        """
        pass

    def reset(self):
        """
        Reset the environment to a (random) initial position.
        :return:
        """
        pass

    def get_demonstrations(self):
        """
        Retrieve the matrix containing the context and the movement parameters stored for this task
        """
        pass

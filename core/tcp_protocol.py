import socket
from pickle import dumps, loads

HEADERSIZE = 4
CHUNK_SIZE = 16


class HyperSocket:

    def __init__(self, conn):
        self._conn = conn

    def receive_all(self):
        ret = b''
        msg = self._conn.recv(HEADERSIZE)
        print("received header", msg[:HEADERSIZE])
        msglen = int.from_bytes(msg[:HEADERSIZE], byteorder='big')
        print("new msg len:", msglen)

        remaining_bytes = msglen
        while True:
            if remaining_bytes >= CHUNK_SIZE:
                ret += self._conn.recv(CHUNK_SIZE)
                remaining_bytes -= CHUNK_SIZE
            elif remaining_bytes == 0:
                break
            else:
                ret += self._conn.recv(remaining_bytes)
                break

        ret = loads(ret)
        print("received", ret)
        return ret

    def send_all(self, msg):
        byte_message = dumps(msg)
        header = len(byte_message).to_bytes(4, byteorder='big')
        print("sending msg len", len(byte_message))
        print("sent header", header)
        packet = header + byte_message
        self._conn.send(packet)
        print("sent", msg)

    def _send_cmd(self, cmd: str):
        self.send_all(cmd)

    def _send_value(self, value):
        self.send_all(value)

    def _wait_done(self):
        return self._wait_cmd("OK")

    def _wait_cmd(self, cmd):
        cmd_rcv = self.receive_all()
        if cmd_rcv == cmd:
            return True
        else:
            raise Exception("Error in communication. Expected %s, Received %s." % (cmd, cmd_rcv))

    def _read_value(self):
        data_rcv = self.receive_all()
        return data_rcv


class Server(HyperSocket):

    def __init__(self, port: int):
        self._port = port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("", port))
        self.socket.listen()

        conn, self._address = self.socket.accept()
        print("Connection accepted", conn)

        super().__init__(conn)

    def send_context_request(self):
        self._send_cmd("CONTEXT")
        return self._read_value()

    def read_context_dim(self):
        self._send_cmd("CONTEXT_DIM")
        return self._read_value()

    def send_movement(self, duration, weigths):
        data_send = {"duration": duration,
                     "weights": weigths}
        self._send_cmd("MOVEMENT")
        self._send_value(data_send)
        received = self._read_value()
        return received["success"], received["dense_reward"]

    def send_reset(self):
        self._send_cmd("RESET")
        return self._wait_cmd("RESET_DONE")

    def wait_demonstration(self):
        self._send_cmd("DEMO")
        return self._read_value()

    def send_n_features(self, n_features):
        self._wait_cmd("N_FEATURES")
        return self._send_value(n_features)


class Client(HyperSocket):

    def __init__(self, ip: str, port: int):
        self._port = port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))

        super().__init__(self.socket)

    def wait_context_request(self):
        return self._wait_cmd("CONTEXT")

    def send_context(self, value):
        return self._send_value(value)

    def send_reward(self, success, dense_reward):
        send = {"success": success,
                "dense_reward": dense_reward}
        self._send_value(send)

    def wait_movement(self):
        self._wait_cmd("MOVEMENT")
        data = self._read_value()
        return data["duration"], data["weights"]

    def wait_reset(self):
        return self._wait_cmd("RESET")

    def reset_ack(self):
        self._send_cmd("RESET_DONE")

    def wait_demonstration_request(self):
        return self._wait_cmd("DEMO")

    def send_demonstration(self, data):
        self._send_value(data)

    def wait_context_dim_request(self):
        self._wait_cmd("CONTEXT_DIM")

    def send_context_dim(self, value):
        self._send_value(value)

    def read_n_features(self):
        self._send_cmd("N_FEATURES")
        return self._read_value()

from core.lab_connection import TCPServerExample
from core.config import config
server = TCPServerExample(5050, 3)
server.run()
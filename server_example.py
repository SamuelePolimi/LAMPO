from core.lab_connection import TCPClientExample
from sshtunnel import SSHTunnelForwarder


with SSHTunnelForwarder(
        ("kay.ias.informatik.tu-darmstadt.de", 22),
        ssh_username="tosatto",
        ssh_pkey="/home/samuele/.ssh/id_rsa_tosatto",
        remote_bind_address=("130.83.164.60", 5056),
        local_bind_address=("127.0.0.1", 5050)
) as tunnel:
    client = TCPClientExample("127.0.0.1", 5050, 3)
    client.run()

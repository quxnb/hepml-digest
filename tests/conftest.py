import socket

import pytest


@pytest.fixture(autouse=True)
def block_live_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must use fake services, not live network connections")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

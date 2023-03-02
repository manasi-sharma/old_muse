import socket
from typing import List


def latest_chunk_from_list(msg_tokens: List[str], seq_len: int, start_item='s'):
    last_start_item = len(msg_tokens)
    for idx in reversed(range(len(msg_tokens))):
        if msg_tokens[idx] == start_item:
            last_start_item = idx
            break

    # .... (2 3 4 1 5) <s> 1 3 ]
    msg_n = msg_tokens[last_start_item - seq_len: last_start_item]
    # .... 2 3 4 1 5 <s> (1 3) ]
    msg_n_plus_1 = msg_tokens[last_start_item + 1:]

    if len(msg_n) < seq_len and len(msg_n_plus_1) < seq_len:
        return None
    else:
        return msg_n_plus_1 if len(msg_n_plus_1) == seq_len else msg_n


def check_ssh(server_ip, port=22):
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect((server_ip, port))
    except Exception as ex:
        # not up, log reason from ex if wanted
        return False
    else:
        test_socket.close()
    return True

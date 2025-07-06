import os
import socket
from typing import List

import requests
from requests.auth import HTTPBasicAuth


from typing import Union, Dict, Any

JSONPrimitive = Union[str, int, float, bool, None]
JSONType = Union[JSONPrimitive, List["JSONType"], Dict[str, "JSONType"]]


def call_predict(
    images: List[os.PathLike],
    api_url: str,
    api_user=None,
    api_password=None,
) -> (JSONType, Dict[str, Any]):
    files = []
    for image_path in images:
        files.append(("image", open(image_path, "rb")))

    post_params = {}

    try:
        request = requests.post(
            api_url,
            files=files,
            auth=HTTPBasicAuth(api_user, api_password)
            if api_user is not None
            else None,
            timeout=(5, 60),
            data=post_params,
        )
    except:
        # TODO:
        raise
        pass

    if request.status_code >= 400:
        print(request)
    return request.json(), post_params


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a service is running on a given host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

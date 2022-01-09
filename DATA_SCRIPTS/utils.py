import json


def config_validator(json_path=None) -> dict:
    with open(json_path) as f:
        json_data = json.load(f)
    if 'host' not in json_data:
        json_data['host'] = "127.0.0.1"
    if 'port' not in json_data:
        json_data['port'] = 2000
    if 'worker_threads' not in json_data:
        json_data['worker_threads'] = 0
    if 'network_timeout' not in json_data:
        json_data['network_timeout'] = 10.0
    if 'width' not in json_data:
        json_data['width'] = 1280
    if 'height' not in json_data:
        json_data['height'] = 720
    if 'role_name' not in json_data:
        json_data['role_name'] = "hero"
    if 'vehicle_filter' not in json_data:
        json_data['vehicle_filter'] = "vehicle.*"
    if 'autopilot' not in json_data:
        json_data['autopilot'] = 0
    if 'gamma' not in json_data:
        json_data['gamma'] = 2.2
    return json_data
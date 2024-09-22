#################################################################################################################
# Waymo celery configuration
#################################################################################################################

import os
from kombu import Exchange, Queue


APP_NAME = "waymo"
_queue_name = f"{APP_NAME}_queue"
_exchange_name = f"{APP_NAME}_tasks"
_routing_key = f"{APP_NAME}_tasks"

base_url = os.getenv('WAYMO_CELERY_BASE_URL', 'redis://redis:6379/')
database_number = os.getenv('WAYMO_CELERY_DATABASE_NUMBER', '0')
broker_url = f"{base_url}{database_number}"
result_backend = f"{base_url}{database_number}"
task_queues = (
    Queue(
        _queue_name,
        exchange=Exchange(_exchange_name, type="direct"),
        routing_key=_routing_key
    ),
)
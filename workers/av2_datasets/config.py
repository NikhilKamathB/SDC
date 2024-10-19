#################################################################################################################
# AV2 celery configuration
#################################################################################################################

import os
from datetime import datetime
from kombu import Exchange, Queue


APP_NAME = "av2"
QUEUE_NAME = f"{APP_NAME}_queue"
_exchange_name = f"{APP_NAME}_tasks"
_routing_key = f"{APP_NAME}_tasks"

base_url = os.getenv('AV2_CELERY_BASE_URL', 'redis://redis:6379/')
database_number = os.getenv('AV2_CELERY_DATABASE_NUMBER', '0')
broker_url = f"{base_url}{database_number}"
result_backend = f"{base_url}{database_number}"
task_queues = (
    Queue(
        QUEUE_NAME,
        exchange=Exchange(_exchange_name, type="direct"),
        routing_key=_routing_key
    ),
)

LOG_DIR = os.getenv("AV2_CELERY_LOG_DIR", "/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S')}_celery_av2_worker.log")

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[%(levelname)s] || %(asctime)s || %(process)d || %(name)s || %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'rich.logging.RichHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOG_FILE,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'tasks': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

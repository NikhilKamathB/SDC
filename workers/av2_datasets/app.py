#################################################################################################################
# AV2 worker app
#################################################################################################################

import config
from celery import Celery
from logging.config import dictConfig
from celery.signals import setup_logging

app = Celery(config.APP_NAME)
app.config_from_object(config)
app.autodiscover_tasks(["tasks"])


@setup_logging.connect
def config_loggers(*args, **kwargs):
    dictConfig(config.LOGGING)

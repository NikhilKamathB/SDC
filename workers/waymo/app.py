#################################################################################################################
# Waymo worker app
#################################################################################################################

import config
from celery import Celery

app = Celery(config.APP_NAME)
app.config_from_object(config)
app.autodiscover_tasks(["tasks"])
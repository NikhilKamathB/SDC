#################################################################################################################
# Vizualization tasks - Waymo specific visualization tasks.
#################################################################################################################

from celery import shared_task


@shared_task(name="test")
def test():
    return "Hello, World!"

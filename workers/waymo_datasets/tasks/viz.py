#################################################################################################################
# Vizualization tasks - Waymo specific visualization tasks.
#################################################################################################################

from celery import shared_task


@shared_task(name="viz.waymo_open_dataset")
def viz_waymo_open_dataset_tf_record(path: str):
    """Visualize Waymo Open Dataset TFRecord."""
    pass
    

@shared_task(name="test")
def test():
    """Test task."""
    return "Hello, World!"

if __name__ == "__main__":
    viz_waymo_open_dataset_tf_record.delay("temp")
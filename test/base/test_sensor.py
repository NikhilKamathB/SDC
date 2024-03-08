##########################################################################################################
# Define test suit for the Sensor class
##########################################################################################################

import carla
import pytest
from typing import Callable
from test.utils import init_client
from src.base.vehicle import Vehicle
from src.base.sensor import Camera_RGB, Camera_Depth


class TestSensor:

    def test_rgb_camera_spawn(self, init_client: Callable):
        """
        Test the spawning of the RGB camera
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            camera = Camera_RGB(world, parent=vehicle.actor)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_depth_camera_spawn(self, init_client: Callable):
        """
        Test the spawning of the depth camera
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            camera = Camera_Depth(world, parent=vehicle.actor)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_rgb_camera_with_transform_spawn(self, init_client: Callable):
        """
        Test the spawning of the RGB camera with transform
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            location = carla.Location(x=10, y=-10, z=0.5)
            rotation = carla.Rotation(yaw=180, pitch=20, roll=10)
            camera = Camera_RGB(world, location=location, rotation=rotation, parent=vehicle.actor)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_depth_camera_with_transform_spawn(self, init_client: Callable):
        """
        Test the spawning of the depth camera with transform
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            location = carla.Location(x=10, y=-10, z=0.5)
            rotation = carla.Rotation(yaw=180, pitch=20, roll=10)
            camera = Camera_Depth(world, location=location,
                                rotation=rotation, parent=vehicle.actor)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_rgb_camera_with_transform_and_kwargs_spawn(self, init_client: Callable):
        """
        Test the spawning of the RGB camera with transform
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            location = carla.Location(x=10, y=-10, z=0.5)
            rotation = carla.Rotation(yaw=180, pitch=20, roll=10)
            camera = Camera_RGB(world, location=location, rotation=rotation, parent=vehicle.actor,
                                fov=120.0, image_size_x=1281, image_size_y=721)
            assert camera.actor.attributes['fov'] == str(120.0)
            assert camera.actor.attributes['image_size_x'] == str(1281)
            assert camera.actor.attributes['image_size_y'] == str(721)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_depth_camera_with_transform_and_kwargs_spawn(self, init_client: Callable):
        """
        Test the spawning of the depth camera with transform
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            vehicle = Vehicle(world)
            location = carla.Location(x=10, y=-10, z=0.5)
            rotation = carla.Rotation(yaw=180, pitch=20, roll=10)
            camera = Camera_Depth(world, location=location, rotation=rotation, parent=vehicle.actor,
                                fov=120.0, image_size_x=1281, image_size_y=721)
            assert camera.actor.attributes['fov'] == str(120.0)
            assert camera.actor.attributes['image_size_x'] == str(1281)
            assert camera.actor.attributes['image_size_y'] == str(721)
            camera.destroy()
            vehicle.destroy()
        except Exception as e:
            pytest.fail(str(e))

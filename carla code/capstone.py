#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import myImport
    from matplotlib import pyplot as plt
    import tensorflow as tf
except ImportError:
    raise RuntimeError('Error in loading myImport')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, model, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def tf_image(image, model, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    image = array[:, :, ::-1]

    resize = (256, 256)
    image_tf = tf.image.convert_image_dtype(image, tf.uint8)
    image_tf = tf.image.resize(image_tf, resize, method='nearest')
    # print(np.shape(image))
    pred = model(np.expand_dims(image_tf, 0), training=False)[0]
    pred_num = pred.numpy()

    t1, t2, b1, b2 = [100, 150], [156, 150], [0, 240], [256, 240]
    birdeyeview = myImport.wrapping(pred_num, t1, t2, b1, b2)

    return image, pred, birdeyeview


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    # display = pygame.display.set_mode(
    #     (800, 600),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    figure, ax = plt.subplots(figsize=(15, 9))

    model, height, width, name = myImport.loadModel(0)
    t1, t2, b1, b2 = [100, 150], [156, 150], [0, 240], [256, 240]
    # model = 0

    # initial data
    f2l = myImport.init_interpolate()
    f2r = myImport.init_interpolate()
    load_2l = myImport.init_interpolate()
    load_2r = myImport.init_interpolate()
    plot_x = np.arange(height)

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.6, z=1.7)),#carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        # camera_semseg = world.spawn_actor(
        #     blueprint_library.find('sensor.camera.semantic_segmentation'),
        #     carla.Transform(carla.Location(x=1.6, z=1.7)),#carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #     attach_to=vehicle)
        # actor_list.append(camera_semseg)



        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=5) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)
                image, pred, birdeyeview = tf_image(image_rgb, model)
                line_bev, line_load_bev = myImport.preprocessing_interpolate(pred, height, width)

                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('raw image', fontsize=15)

                plt.subplot(1, 3, 2)
                plt.imshow(pred)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('segmentation', fontsize=15)
                [plt.scatter(point[0], point[1]) for point in [t1, t2, b1, b2]]

                plt.subplot(1, 3, 3)
                plt.imshow(birdeyeview)

                f2l, f2r = myImport.make_interpolate(f2l, f2r, line_bev)
                load_2l, load_2r = myImport.make_interpolate(load_2l, load_2r, line_load_bev)

                plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), f2l(plot_x))),
                         list(filter(lambda x: (f2l(x) < 256) & (f2l(x) >= 0), plot_x)), '--',
                         color='k')  # birdeyeview에 적용시
                plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), f2r(plot_x))),
                         list(filter(lambda x: (f2r(x) < 256) & (f2r(x) >= 0), plot_x)), '--', color='w')

                plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), load_2l(plot_x))),
                         list(filter(lambda x: (load_2l(x) < 256) & (load_2l(x) >= 0), plot_x)), '--', color='w')
                plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), load_2r(plot_x))),
                         list(filter(lambda x: (load_2r(x) < 256) & (load_2r(x) >= 0), plot_x)), '--', color='w')
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('birdeyeview', fontsize=15)

                plt.draw()
                plt.pause(0.001)
                figure.clear()
                # print(type(snapshot))
                # print(type(image_rgb))

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                # draw_image(display, image_rgb, model)
                # print(type(image_rgb))
                # figure.clear()
                # draw_image(display, image_semseg, blend=True)
                # display.blit(
                #     font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                #     (8, 10))
                # display.blit(
                #     font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                #     (8, 28))
                # pygame.display.flip()


    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
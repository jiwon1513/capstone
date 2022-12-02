#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
sys.path.append('C:/Users/mycom/git/ufld')
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import ColorConverter as cc
#from myLib import *
from utils import *

import random
import time
import numpy as np
import cv2
import pygame
import math
import weakref
import torch
from matplotlib import pyplot as plt
import time
import queue

IM_WIDTH = 1280
IM_HEIGHT = 720
Yaw = 90
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

IMAGE0 = []
LANE0 = []


weights = 'models/yolopv2.pt'
device = '0'

model = torch.jit.load(weights)
device = select_device(device)
model = model.to(device)
model.half()
model.eval()

stride = 32
imgsz = 640

#yolov5
'''
class ObjectDetection:
    # YouTube 동영상에 YOLOv5 구현

    def __init__(self, url, out_file):
        # 객체 생성 시 호출
        # url: 예측 대상 YouTube URL
        # out_file: 유효한 출력 파일 이름 *.avi
        
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        # YOLOv5 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self):
        # 인스턴스 생성 시 호출; 프레임 단위로 비디오 로드
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)




def class_to_label(x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return model.names[int(x)]
'''





def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color img인 경우 :
        color = color3
    else:                   # 흑백 img인 경우 :
        color = color1

    cv2.fillPoly(mask, vertices, color)
    # vertices에 정한 점들로 이뤄진 mask 영역(ROI 설정 부분)을 color로 채움
    ROI_image = cv2.bitwise_and(img, mask)
    # 이미지와 color로 채워진 ROI를 합침
    return ROI_image

def find_lane(roi):
    
    left = []
    right = []
    for i in range(400,len(roi)):
        for j in range(0,580):
            if roi[i][580 - j] == 1:
                left.append([i,580 - j])
                break
        for j in range(0,1280-580):
            if roi[i][580 + j] == 1:
                right.append([i,580 + j])
                break
    
    for i in range(len(left)):
        x = left[i][1]
        y = left[i][0]
        left[i][0] = x
        left[i][1] = y
        cv2.circle(roi,(x,y),10,(255,255,255),-1)
    for i in range(len(right)):
        x = right[i][1]
        y = right[i][0]
        right[i][0] = x
        right[i][1] = y
        cv2.circle(roi,(x,y),10,(100,100,100),-1)
    return left,right        
def distance(p1,p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    length = math.sqrt((x*x) + (y*y))
    return length
 
def findR(lane_points):
   
    a = lane_points[0] 
    b = lane_points[-1]

    x = b[0] - a[0] 
    y = b[1] - a[1]
    w = math.sqrt((x*x) + (y*y))
   
    m = -x/y
    M = [(a[0]+b[0])//2 ,(a[1]+b[1])//2]
    
    
   
    C = 1000
    z = 0
    for i in range (len(lane_points)):
        if lane_points[i][0] - M[0] != 0:
            c = abs(((lane_points[i][1] - M[1]) / (lane_points[i][0] - M[0])) - m)
       
            if c < C:
            
                C = c
                z = i
        else:
            z= i
            break
    h = distance(M,lane_points[z])
    try:
        R = h/2 + (w*w)/(8*h)
    except:
        R = 1000000000
    #실제 좌표의 차이로 나오는 차선의 폭과 실제 차선의 폭 3.5를 비례식을 이용해서 다시 구하기
    RR = 3.5*R/1280
    #cv2.circle(img,lane_points[z],5, (0,255,255), -1)
    return RR

def steerangle(lane_points1, lane_points2):
    R1 = findR(lane_points1)
    R2 = findR(lane_points2)
    #print("r1, r2" ,R1,R2)
    L = 2.89
    T = 1.64
    R = (R1+R2)/2
    thetain  = math.degrees(math.atan(L/(R-(T/2))))
    thetaout = math.degrees(math.atan(L/(R+(T/2))))
    #print(thetain)
    return thetain, thetaout



   
def process_img(image):
    start = time.time()
   
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    frame = i2[:, :, :3]
    # output_img, points = lane_detector.detect_lanes(frame)
    #output_img,l1,l2 = lane_detector.detect_lanes(frame) 
    
    global IMAGE0
    global LANE0
    img0 = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    _, _, ll = model(img)
    ll_seg_mask = lane_line_mask(ll)
   
    

    '''
    v = np.array([[(0,720),(0,480),(510,450),(800,450),(1280,480),(1280,720)]], dtype=np.int32)     
    roi = region_of_interest(ll_seg_mask, v)
    '''  
    roi = ll_seg_mask
    IMAGE0 = img0
    
    
    l1,l2 = find_lane(roi)
    #l1 = left lane ,l2 = right lane
    LANE0 = roi
    
    try:
        
        left = np.array(l1)
        right = np.array(l2)
        
        left  = np.flipud(left)
        right  = np.flipud(right)
        #a = np.where((arr[:,1] <510) & (arr[:,1]>485))
        
        lf = left[-1][0]
        rf = right[-1][0]
        #왼차선 오른차선 가장 아래점의 x 좌표
        center =639
        center1 = (lf+rf)//2
        #print(center1)
        if center -center1 >0 :
            
            
            thetain, thetaout = steerangle(left,right)
            
            angle = -(thetain +thetaout)/140
        else:
            thetain, thetaout = steerangle(left,right)
            angle = (thetain +thetaout)/140
             
        '''    
        cp = [(arr[a][0][0] + arr1[b][0][0])//2 , (arr[a][0][1] + arr1[b][0][1])//2]
        cv2.circle(output_img, cp,5, (0,255,0), -1)
        c[0] = c[1]
        c[1] = cp[0]
        '''
    except:
        angle = 0
             
        
        
    
    
   
    
    '''
    if left == True:
        angle = -angle
        
        
    elif right == True:
        
        angle = angle
    else:
        thetain, thetaout = steerangle(l1,l2,output_img)
        angle = -(thetain +thetaout)/140    '''

    
 
    if angle > 1:
        angle  =1
    elif angle <-1:    
        angle = -1
    else:
        angle = angle
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0)) 
    st = round(angle,3)
    x = abs(st) 
    xx= 0
    if x > 0.03:                  
        print("Steer :",st)
        
        if x > 0.23 :
            if x > 0.35 :
                if x > 0.55:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= st))
                    time.sleep(0.2)
                    xx =0.2
                    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= 0))    
                else :
                    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= st))
                    time.sleep(0.13)
                    xx =0.13
                    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= 0))
            
            else:    
                vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= st))
                time.sleep(0.08)
                xx = 0.08
                vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= 0))   
        else:
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= st))
            time.sleep(0.04)
            xx = 0.04
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= 0))    
    else:
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
    
    end = time.time()
    '''
    tt = end - start-xx
    tt1 = end - start
    print('time:', tt)
    print('time1:', tt1)'''
    
    #vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer= 0))
      
        
    #if sp >30  :
        #vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer=0))
    
    
    

    
  
    #yolov5
    '''
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame = frame.copy()
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
           #cv2.putText(frame,class_to_label(labels[i]) + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr, 2)
            cv2.putText(frame,class_to_label(labels[i]),(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr, 2)
            
    '''         
    # cv2.imshow("",frame)
    # cv2.waitKey(1)
    # im_rgb = cv2.cvtColor(predict_image.imgs[0], cv2.COLOR_BGR2RGB) # Because of OpenCV reading images as BGR
    # cv2.imshow("",im_rgb)


def process_img1(image, lane_detector):
    
    
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    frame = i2[:, :, :3]
    # output_img, points = lane_detector.detect_lanes(frame)
    output_img,l1,l2 = lane_detector.detect_lanes(frame)
    #print(point)
    global IMAGE1
    
    IMAGE1 = output_img
    global left
    try:
        arr = np.array(l1)
        a = np.where((arr[:,1] <510) & (arr[:,1]>485))

        arr1 = np.array(l2)
        b = np.where((arr1[:,1] <510) & (arr1[:,1]>485))          

             
    
        cp = [(arr[a][0][0] + arr1[b][0][0])//2 , (arr[a][0][1] + arr1[b][0][1])//2]
        cv2.circle(output_img, cp,5, (0,255,0), -1)
        
        left = True
        
    except:
        left = False
   

   
    
def process_img2(image, lane_detector):
    
   
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    frame = i2[:, :, :3]
    # output_img, points = lane_detector.detect_lanes(frame)
    output_img,l1,l2 = lane_detector.detect_lanes(frame)
    global IMAGE2
    
    IMAGE2 = output_img 
    global right
    #print(point)
    try:
        arr = np.array(l1)
        a = np.where((arr[:,1] <510) & (arr[:,1]>485))

        arr1 = np.array(l2)
        b = np.where((arr1[:,1] <510) & (arr1[:,1]>485))          

             
    
        cp = [(arr[a][0][0] + arr1[b][0][0])//2 , (arr[a][0][1] + arr1[b][0][1])//2]
        cv2.circle(output_img, cp,5, (0,255,0), -1)
        
        right = True
        
        
    except:
        right = False

      
    
def convertimage(image):
    image.convert(cc.CityScapesPalette)
    i = np.reshape(np.copy(image.raw_data),(IM_HEIGHT,IM_WIDTH,4))
    if (i[400][100] == np.array([128,64,128,255])).min():
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
    
    v = vehicle.get_velocity()
    sp = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    if sp >30  :
        vehicle.apply_control(carla.VehicleControl(throttle = 0.5, steer=0))
   
    #cv2.imshow("", i)
    #cv2.waitKey(1)

class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        
      
            
        self.hud = hud
        self.player = None    

   

 
       

       



    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
# Render object to keep and pass the PyGame surface


class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))
    

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))

        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
     
        v = vehicle.get_velocity()
        c = vehicle.get_control()
       
       
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',         
            
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
   

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
       

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)






    
actor_list = []
camera_list = []
figure = plt.figure()
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)

    world = client.get_world()
    '''
    model_path = "C:/Users/mycom/git/ufld/models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = True
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)
    '''
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

   
    x, y, z, degree = 106.3, -3, 1, 271
    #커브 출발
    
    #x, y, z, degree = 105.7, 97, 1, 270
    #직선출발
    
    spawn_point = carla.Transform(carla.Location(x, y, z), carla.Rotation(0, degree, 0))
    #spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)

    #physics_control = vehicle.get_physics_control()

    # For each Wheel Physics Control, print maximum steer angle
    #for wheel in physics_control.wheels:
        #print (wheel.max_steer_angle)    
    #vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)
    
    #print(carla.VehicleWheelLocation)
    #pygame camera
    time.sleep(0.5)
    camera_init_trans = carla.Transform(carla.Location(x=-5,z =3),carla.Rotation(pitch=-20))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp,camera_init_trans, attach_to=vehicle)
    camera_list.append(camera)

# Start camera with PyGame callback
    camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
    global image_w
    global image_h
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    
    
    

# Instantiate objects for rendering and vehicle control
    renderObject = RenderObject(image_w, image_h)

    # spawn the sensor and attach to vehicle.
    blueprint = blueprint_library.find('sensor.camera.rgb')
    
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    # change the dimensions of the image
    '''
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')
    blueprint.set_attribute('sensor_tick','0.3')'''
    '''
    spawn_point1 = carla.Transform(carla.Location(x=2.5 ,y = 2.5 ,z=0.7), carla.Rotation(yaw = 45))
    sensor1 = world.spawn_actor(blueprint, spawn_point1, attach_to=vehicle)
    actor_list.append(sensor1)
    sensor1.listen(lambda image: process_img1(image, lane_detector))
    

    spawn_point2 = carla.Transform(carla.Location(x=2.5 ,y = -2.5 ,z=0.7), carla.Rotation(yaw = -45))
    sensor2 = world.spawn_actor(blueprint, spawn_point2, attach_to=vehicle)
    actor_list.append(sensor2)
    sensor2.listen(lambda image: process_img2(image, lane_detector)) 
   '''
    spawn_point = carla.Transform(carla.Location(x=2.5,z=0.7))
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    # add sensor to list of actors
    actor_list.append(sensor)
    # do something with this sensor
    
    sensor.listen(lambda image: process_img(image))
    
    
    
   
   

    
    '''
    #segcamera 
    blueprint = blueprint_library.find('sensor.camera.semantic_segmentation')
    
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')
    blueprint.set_attribute('sensor_tick','0.3')

    # Adjust sensor relative to vehicle
    
    
    
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    # spawn the sensor and attach to vehicle.
    sensor2 = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    # add sensor to list of actors
    actor_list.append(sensor2)
    # do something with this sensor
    sensor2.listen(lambda image: convertimage(image))   '''  
       
    pygame.init()
    hud = HUD(100,480)
    gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
    gameDisplay.fill((0,0,0))
    clock = pygame.time.Clock()
    
    
    
    

    crashed = False

    while not crashed:
    # Advance the simulation time
        
        world = World(world, hud)
        
        
        gameDisplay.blit(renderObject.surface, (0,0))
        hud.tick(world, clock)
        hud.render(gameDisplay)
        pygame.display.flip()

        if IMAGE0 != []:
            plt.imshow(IMAGE0)
            plt.imshow(LANE0, alpha=0.5)
            
            
           
            
            
            '''
            plt.subplot(1, 3, 3)
            plt.imshow(IMAGE1)
            plt.subplot(1, 3, 1)
            plt.imshow(IMAGE2)
            '''
            plt.draw()
           
            plt.pause(0.1)
            
            figure.clear()
            
   
        for event in pygame.event.get():
    # If the window is closed, break the while loop
            if event.type == pygame.QUIT:
                crashed = True
    # Process the current control state    
       
# Stop camera and quit PyGame after exiting game loop


    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
   
    
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    #time.sleep(5)


   
finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    for camera in camera_list:
        camera.stop()
        camera.destroy()
    pygame.quit()        
    print('done.')
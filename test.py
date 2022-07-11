import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3 '

# from predict import *
from test2 import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

main_path = 'E:/dataset/'
image_path = main_path + "test/RGB/"

image_list = []
image_temp = os.listdir(image_path)
image_list.extend([image_path + i for i in image_temp])

num = 0
model, height, width, name = loadModel(num)
# model.summary()
figure, ax = plt.subplots(figsize=(9, 4))
n = 0
maxTime = 0.
meanTime = 0.

# birdeyeview ROI
# t1, t2, b1, b2 = [100, 150], [156, 150], [0, 240], [256, 240]
t1, t2, b1, b2 = [width-154, height-106], [width-102, height-106], [0, height-16], [width, height-16]

# interpolate initial value
x = [10, 60, 110]
y = [240, 195, 150]
# y = [width-16, width-61, width-116]
# f1 = interpolate.interp1d(x, y)
f1l = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
f1r = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
f2l = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
f2r = interpolate.InterpolatedUnivariateSpline(x, y, k=2)

load_1l = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
load_1r = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
load_2l = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
load_2r = interpolate.InterpolatedUnivariateSpline(x, y, k=2)
plot_x = np.arange(height)
# plt.plot(plot_x, f1(plot_x))
# plt.show()

for n in range(len(image_list)):
    start = time()
    pred, image = predict(model, height, width, image_list[n])
    # plt.imshow(pred[0])
    # plt.show()

    # tensorflow에서 제공하는 crop 기능
    # image = cv2.imdecode(pred, cv2.IMREAD_COLOR)
    # box = [[100, 150], [40, 256], [140, 150], [200, 256]]
    # birdeyeview = tf.image.crop_and_resize(pred, boxes=[[0, 0, 1, 1]], crop_size=[256, 256], box_indices=[0])
    # 사각형으로만 잘리는 한계

    pred_num = pred.numpy()

    B = tf.math.argmax(pred_num[0], 2)    # segmenstation에 적용시

    birdeyeview = wrapping(pred_num[0], t1, t2, b1, b2)
    image_bev = tf.math.argmax(birdeyeview, 2)    # birdeyeview에 적용시
    image_bev = tf.keras.backend.eval(image_bev)    # birdeyeview에 적용시

    line_left_top, line_left_bottom, line_right_top, line_right_bottom = \
        [0, 0], [0, height], [width, 0], [width, height]

    # 값이 2인 곳 위치 출력
    line = tf.where(tf.equal(B, 2))    # segmenstation에 적용시
    line_bev = tf.where(tf.equal(image_bev, 2))    # birdeyeview에 적용시
    # print(line)

    line = line.numpy()  # x값 기준 중복값 처리 및 보간법 적용을 위한 전처리
    line_bev = line_bev.numpy()  # x값 기준 중복값 처리 및 보간법 적용을 위한 전처리

    f1l, f1r = make_interpolate(f1l, f1r, line)
    f2l, f2r = make_interpolate(f2l, f2r, line_bev)

    line_load = tf.where(tf.equal(B, 1)).numpy()    # segmenstation에 적용시
    line_load_bev = tf.where(tf.equal(image_bev, 1)).numpy()    # birdeyeview에 적용시

    load_1l, load_1r = make_interpolate(load_1l, load_1r, line_load)
    load_2l, load_2r = make_interpolate(load_2l, load_2r, line_load_bev)

    print(f'{n}th predict time: {time() - start}')
    if (time() - start) > maxTime:
        maxTime = time() - start
    meanTime += time() - start

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(pred[0])
    [plt.scatter(point[0], point[1]) for point in [t1, t2, b1, b2]]

    plt.plot(list(filter(lambda x: (x < int(width/2)) & (x >= 0), f1l(plot_x))),
             list(filter(lambda x: (f1l(x) < int(width/2)) & (f1l(x) >= 0), plot_x)), '--', color='k')    # segmenstation에 적용시
    plt.plot(list(filter(lambda x: (x < int(width/2)) & (x >= 0), load_1l(plot_x))),
             list(filter(lambda x: (load_1l(x) < int(width/2)) & (load_1l(x) >= 0), plot_x)), '--', color='w')
    plt.plot(list(filter(lambda x: (x > int(width/2)) & (x >= 0), load_1r(plot_x))),
             list(filter(lambda x: (load_1r(x) > int(width/2)) & (load_1r(x) >= 0), plot_x)), '--', color='w')

    plt.subplot(1, 3, 3)
    plt.imshow(birdeyeview)
    plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), f2l(plot_x))),
             list(filter(lambda x: (f2l(x) < 256) & (f2l(x) >= 0), plot_x)), '--', color='k')    # birdeyeview에 적용시
    plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), load_2l(plot_x))),
             list(filter(lambda x: (load_2l(x) < 256) & (load_2l(x) >= 0), plot_x)), '--', color='w')
    plt.plot(list(filter(lambda x: (x < 256) & (x >= 0), load_2r(plot_x))),
             list(filter(lambda x: (load_2r(x) < 256) & (load_2r(x) >= 0), plot_x)), '--', color='w')
    plt.draw()
    plt.pause(0.01)
    figure.clear()

print(f'predict total time: {meanTime}')
print(f'predict mean time: {meanTime/int(len(image_list))}')
print(f'predict max time: {maxTime}')

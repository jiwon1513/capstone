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
for n in range(len(image_list)):
    start = time()
    pred, image = predict(model, height, width, image_list[n])
    # plt.imshow(pred[0])
    # plt.show()

    # image = cv2.imdecode(pred, cv2.IMREAD_COLOR)
    t1, t2, b1, b2 = [100, 150], [156, 150], [0, 240], [256, 240]
    # box = [[100, 150], [40, 256], [140, 150], [200, 256]]

    # birdeyeview = tf.image.crop_and_resize(pred, boxes=[[0, 0, 1, 1]], crop_size=[256, 256], box_indices=[0])
    pred_num = pred.numpy()
    birdeyeview = wrapping(pred_num[0], t1, t2, b1, b2)
    # B = tf.math.argmax(pred[0], 2)
    B = tf.math.argmax(birdeyeview, 1)
    C = tf.keras.backend.eval(B)

    line_left_top, line_left_bottom, line_right_top, line_right_bottom = \
        [0, 0], [0, 256], [256, 0], [256, 256]

    line = tf.where(tf.equal(C, 2))
    print(line)

    print(f'{n}th predict time: {time() - start}')
    if (time() - start) > maxTime:
        maxTime = time() - start
    meanTime += time() - start

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(pred[0])
    [plt.scatter(point[0], point[1]) for point in [t1, t2, b1, b2]]
    plt.subplot(1, 3, 3)
    plt.imshow(birdeyeview)
    plt.draw()
    plt.pause(0)
    figure.clear()

print(f'predict total time: {meanTime}')
print(f'predict mean time: {meanTime/int(len(image_list))}')
print(f'predict max time: {maxTime}')

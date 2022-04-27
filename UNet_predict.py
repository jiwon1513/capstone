import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

file_path = 'D:/download/dataB/'

test_images=np.load(file_path+'test_images.npy')
test_masks=np.load(file_path+'test_masks.npy')

model_name_list = ['UNet', 'UNet_mini', 'ResNet_UNet', 'PSPNet_UNet']
model_name = model_name_list[0]
# set directory
os.chdir(file_path+'results/')
filepath = 'road-seg-model.h5'
loaded_model=load_model(filepath)
# loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(round(loaded_model.evaluate(x=test_images,y=test_masks,batch_size=16)[1],4))

# test data num
NUMBER = 1
my_preds = loaded_model.predict(np.expand_dims(test_images[NUMBER], 0))
my_preds = my_preds.flatten()
my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])

fig = plt.figure()
rows = 1
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(my_preds.reshape(600,800))
ax1.set_title('prediction')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(test_masks[NUMBER].reshape(600, 800))
ax2.set_title('real')
ax2.axis("off")

plt.savefig(file_path + 'results/' + model_name + '_plot.png')
plt.show()
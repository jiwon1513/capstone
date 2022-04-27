# capstone

library
1. python == 8.0
2. tensorflow == 2.4.0
3. tensorflow-gpu == 2.4.0
4. CUDA == 8.0
5. CUDA toolkit == 11.1

data preprocessing
1. .npy convert
2. argumentation  - error
3. resize - eeerrrrrorooorrrrrrrrr label 파일을 어떻게 처리해야하는겨ㅕㅕㅕㅕㅕㅕ

learning segmentation model
1. UNet
2. mini_UNet  - OOM / soluted by update tensorflow & tf-gpu (2.3.0 -> 2.4.0)
3. ResNet_UNet  - input size error / need to resize images and labels
4. PSPNet - input size error / need to resize images and labels
5. ResNet_PSPNet  - input size error / need to resize images and labels

optimization
1. compile
2. callback
3. softmax??

test model
1. test dataset (carla)
2. other dataset (real photo)
3. w/ my photo around me

4.27
- 현재 여러 모델을 사용하여 정확도가 높은 모델을 쓰고싶은 욕심이 있었으나
- 계속되는 에러를 처리하지 못하여 굳이 모델을 추가하는데 애쓸 필요가 있을까싶음
- UNet을 통해 처리하는 방향이 훨씬 빠를것으로 예상됨
- -> UNet optimization 단계를 진행하고 결과 도출을 우선으로 

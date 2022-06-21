# capstone

library
1. python == 8.0
2. tensorflow == 2.4.0
3. tensorflow-gpu == 2.4.0
4. CUDA == 8.0
5. CUDA toolkit == 11.1

data preprocessing
1. .npy convert
2. resize - done - tf.image.resize()

learning segmentation model
1. UNet
2. mini_UNet  - OOM / soluted by update tensorflow & tf-gpu (2.3.0 -> 2.4.0)
3. ResNet_UNet  - input size (224, 224)
4. UNet_PSPNet - decorder shape error
5. ResNet_PSPNet
6. 
optimization
1. compile - adam 쓰는게 무난
2. callback - 크게 안바꿔도 괜찮을듯
3. activation - softmax: muti-Classification / sigmoid: binary-Classification

test model
1. test dataset (carla)
2. other dataset (carla) - I can make it with my desktop

4.27
- 현재 여러 모델을 사용하여 정확도가 높은 모델을 쓰고싶은 욕심이 있었으나
- 계속되는 에러를 처리하지 못하여 굳이 모델을 추가하는데 애쓸 필요가 있을까싶음
- UNet을 통해 처리하는 방향이 훨씬 빠를것으로 예상됨
- -> UNet optimization 단계를 진행하고 결과 도출을 우선으로 

5.9
- UNet을 더 복잡한 모델로 변경
- PSPNet은 제거, ResNet은 보류
- Carla simulator 파일 손상으로 재설치 필요

5.23
- UNet 기반 정확도 측정 - 정확도가 상당히 떨어짐, dataSet 추가 & lower classes 도입 예정
- ResNet 학습성공, PSPNet 학습 재도전 - 여러 모델들을 학습시켜 결과를 비교할 예정
- carla custom control code - 기본 예제 파일에 추가 작성하여 dataset 추출까진 가능, 이후엔 필요한 기능만 넣은 코드로 변경 및 조종 코드 추가 

# capstone

library
1. python == 8.0
2. tensorflow == 2.4.0
3. tensorflow-gpu == 2.4.0
4. CUDA == 8.0
5. CUDA toolkit == 11.1

data preprocessing
1. .npy convert
2. argumentation  - 보류
3. resize - done/error - tf.image.resize() 다시 시도

learning segmentation model
1. UNet - layer 추가 수정
2. mini_UNet  - OOM / soluted by update tensorflow & tf-gpu (2.3.0 -> 2.4.0) / 코드 삭제
3. ResNet_UNet  - input size error / need to resize images and labels - 224, 224 고정 / 보류
4. PSPNet - input size는 맞추었으나 학습이 안됨, 현재 세팅에서 많이 벗어날 것 같음 - 코드 삭제
5. ResNet_PSPNet  - input size error / need to resize images and labels - 코드 삭제

optimization
1. compile - adam 쓰는게 무난
2. callback - 크게 안바꿔도 괜찮을듯
3. activation - softmax: muti-Classification / sigmoid: binary-Classification

test model
1. test dataset (carla)
2. other dataset (real photo) - UNet 모델론 정확성 떨어짐(carla 학습 결과) - 따로 모델링
3. w/ my photo around me

4.27
- 현재 여러 모델을 사용하여 정확도가 높은 모델을 쓰고싶은 욕심이 있었으나
- 계속되는 에러를 처리하지 못하여 굳이 모델을 추가하는데 애쓸 필요가 있을까싶음
- UNet을 통해 처리하는 방향이 훨씬 빠를것으로 예상됨
- -> UNet optimization 단계를 진행하고 결과 도출을 우선으로 

5.9
- UNet을 더 복잡한 모델로 변경
- PSPNet은 제거, ResNet은 보류
- Carla simulator 파일 손상으로 재설치 필요

5.10
- Carla simulation 재설치 완료
- UNet 레이어 추가하여 carla dataset으로 학습
- UNet으로 고정, dataSet 별로 학습하여 정확성 테스트 예정
- Carla code 우선 작업 예정

# Artificial-Bee-Colony-with-YOLOv7-Tiny
combination of YOLOv7 and Artificial Bee Colony(ABC) Algorithm.

# 목적 및 감사
이 저장소는 대한민국 정보처리학회 학부생논문경진대회에 제출할 '물체 탐지에서 Neural Architecture Search 기반 Channel Pruning을 통한 Parameter-Efficiency 개선' 논문의 보충 설명 및 사용된 코드를 제공하기 위해 만들어졌습니다.

또한 Base가 된 코드는 ```https://github.com/lmbxmu/ABCPruner``` 이 레포지토리에서 왔음을 알립니다. 
관련 논문에는 Channel Pruning via Automatic Structure Search(```https://arxiv.org/pdf/2001.08565.pdf```)가 있습니다.

감사합니다.

# 의존성
이 저장소는 사용자가 기존 YOLOv7 코드를 원활하게 학습 및 테스트가 가능한 환경에 있다고 가정하고 짜여진 코드입니다. 
YOLOv7이 원활하게 동작한다면 이 코드도 동작할 것입니다.

# 코드 이식
이 코드는 YOLOv7-Tiny를 목표로 제작되었고 테스트되었습니다. 
YOLOv7의 다른 크기 모델에서는 동작을 확인하지 않았음을 안내드립니다.
1. ```bee_yolov7_tiny.py```파일을 YOLOv7 마스터 폴더 안에 넣어주세요.
2. ```ABCPruner_*.py```파일들을 ```YOLOv7/utils/```에 넣어주세요.

# 실행
다음과 같은 명령어로 실행합니다.
```
python3 bee_yolov7_tiny.py --data_path {YOUR_DATA.yaml} --honey_model {YOUR_PRETRAINED.pt} --job_dir {RESULT_DIR_PATH} --arch {YOUR_ARCHITECTURE} --cfg {YOUR_MODEL_CFG.yaml} --num_epochs {FINAL_MODEL_TRAINING_EPOCH} --gpus {ALWAYS SHOULD BE 0} --calfitness_epoch {HONEYCODE_TESTING_EPOCH} --max_cycle {TOTAL_ABC_CYCLE} --max_preserve {UPBOUND_OF_EACH_LAYER_CHANNEL_NUMBER(means ratio, integer, 2~10)} --food_number {BEE_NUMBER} --food_limit {MAX_TRIAL_LIMIT} --random_rule {RULE_OF_IMPORTING_WEIGHT_FROM_ORIGINAL_MODEL, random_pretrain or l1_pretrain} --hyp {YOUR_YOLOV7_HYP.yaml} --img_size {YOUR_IMG_SIZE} --train_batch_size {YOUR_BATCH_SIZE}
```


실제 예시는 다음과 같습니다.
```
python3 bee_yolov7_tiny.py --data_path ABCPruner/car_rjh_local.yaml --honey_model yolov7-tiny.pt --job_dir ABC_DIR/argo_alpha_5 --arch yolov7-tiny --cfg cfg/training/yolov7-tiny.yaml --num_epochs 30 --gpus 0 --calfitness_epoch 2 --max_cycle 5 --max_preserve 5 --food_number 5 --food_limit 2 --random_rule random_pretrain --hyp data/hyp.scratch.tiny.yaml --img_size 640 --train_batch_size 16
```

# 다른 크기의 YOLOv7 모델을 실행시키고 싶다면:
1. ```bee_yolov7_tiny.py```파일의 첫 부분의 conv_num_cfg 딕셔너리를 찾으세요.
2. 딕셔너리에 원하는 모델의 이름과 그 모델에 포함된 모든 컨볼루션 레이어의 수를
```'NAME': NUM_OF_CONV_LAYER``` 이 형식으로 추가해주세요.
3. cfg옵션과 hyp옵션, honey_model옵션을 원하는 YOLOv7모델의 것으로 입력해주세요.

# 결과 및 실험 환경
![graph](https://github.com/Jochiwon/Artificial-Bee-Colony-with-YOLOv7-Tiny/assets/44675901/03a56fd0-1aab-4429-a780-fdcfb4e67f62)

실험 결과는 위 그림과 같습니다.

U-90%보다 U-80%가 더 파라미터가 큰 이유는 Upbound U는 honey code의 최대값을 제한하는 역할이기 때문에 그렇습니다.

U가 9일때 honey code가 [6, 2, 9]이고 U가 8일때 honey code가 [6, 6, 8]이 될 수도 있습니다.

U를 낮춘다고 결과 모델이 꼭 이전보다 더 작아진다는 보장은 없는 것이죠.

실험에 진행한 실험 환경은 다음과 같습니다.
```
OS: Ubuntu 20.04 LTS
CPU: Intel i5-9400F
GPU: RTX 3090
RAM: 32GB
Baseline Model: YOLOv7-Tiny
Input Img Size: 640*640
Batch Size: 16
Epoch: 30
Pytorch: 2.0.1+cu118
CUDA: 11.8
```


# 설명
3개의 컨볼루션 레이어로 이뤄진 모델이 있다고 가정해봅시다.

이때 각 레이어별로 채널 숫자를 1~10의 비율로 표현한다면 현재 상태는 [10, 10, 10]으로 볼 수 있겠죠.

이것을 코드에서는 honey code라고 부릅니다.

이 코드의 목표는 honey code의 숫자를 조정하여 가장 높은 fitness를 가지는 honey code를 찾아낸 뒤 이 코드로 학습시킨 모델을 저장하는 것입니다.

예를 들어 결과 honey code가 [5, 2, 7]이라고 할 때 이는 첫번째 레이어의 채널 수를 기존 대비 5/10, 두번째 레이어의 채널 수를 기존 대비 2/10, 세번째 레이어의 채널 수를 기존 대비 7/10으로 줄인다는 뜻입니다.

random_rule을 random_pretrain이라 한다면 honey code로 pruning 후 기존 pretrain데이터에서 weight를 랜덤하게 가져오기 때문에 random weight pruning이라 볼 수 있습니다.

ABC 알고리즘의 설명은 아래 링크를 참고 부탁드립니다.

```https://untitledtblog.tistory.com/53```

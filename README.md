# 🎓Attention-Ensemble-Translation🎓
- [RNN Translation](https://github.com/aaajeong/RNN-Translation) - Tensorflow Attention 기계번역 Ensemble  연구 이어서 진행.



#### 📝 연구 주제 : Natural Language Translation 에서 Ensemble 을 사용하면 성능이 좋아질까?


1. **nmt_with_attention.ipynb**

   - [코드](./nmt_with_attention.ipynb)
   - [**nmt_with_attention_Test.ipynb**](./nmt_with_attetion_Test.ipynb)

2. **nmt_with_attetntion_Ensemble_HardVoting.ipynb**

   - [코드](./nmt_with_attetion_Ensemble_HardVoting.ipynb)

   **💥 문제 발견 💥**

3. **nmt_with_attention_Ensemble_SoftVoting.ipynb**

   - [코드](./nmt_with_attetion_Ensemble_SoftVoting.ipynb)

   **💥 문제 발견 💥**

4. **nmt_with_attention_randomly_data.ipynb**

   - [코드](./nmt_with_attention_randomly_data.ipynb)

5. **nmt_with_attention_Ensemble_SoftVoting(2).ipynb**

   - [코드](./nmt_with_attetion_Ensemble_SoftVoting(2).ipynb)

   **💥 문제 발견 💥**

---

#### 💥문제 발견 설명💥

	- [앙상블 보팅 과정에서 모델 동작 확인](https://github.com/aaajeong/NMT-Ensemble-Study/commit/5799477fe58b9194502bbcf8ac0a4f5100d00fdc)
 - 문제 발견
    -  각 모델에서 만든 데이터 딕셔너리 (ex. {1:'start', ..., 36:'cold'}) 형태가 다 다름. 그런  데도 불구하고 prediction 의 결과를 모델 1의 targ_lang 딕셔너리에서 불러오고 있었음.
      	- 👉 모델 2, 3 의 데이터 딕셔너리를 무시한 꼴
    - checkpoint 를 하나만 남겨도 동작함
      	- 👉 checkpoint 를 여러개 만들어서 불러와도 마지막으로 불린 체크포인트만 적용. 따라서 첫번째 모델은 제대로 동작하지만 나머지2,3 모델은 쓰레기값이 들어가고 있었음.
   - **이 두가지 해결해서 다시 앙상블 정확도 확인!!**
     - **새로운 파일 : re-nmt_Ensemble_HardVoting.ipynb , re-nmt_with_Ensemble_SoftVoting.ipynb**

---



#### 🔎 Ensemble 을 이용한 NMT 정확도 확인(Model 3개)

- Model 개수 : 3개

- 트레이닝 데이터 : [spa-eng/spa_for_esb.txt](./spa-eng/spa_for_esb.txt)

- Training Checkpoint : 각 3개의 모델에 대한 파일

  - Model 1 : [training_checkpoints_esb](./3 Models Checkpoints/training_checkpoints_esb)
  - Model 2 : [training_checkpoints_esb 2](./3 Models Checkpoints/training_checkpoints_esb 2)
  - Model 3 : [training_checkpoints_esb 3](./3 Models Checkpoints/training_checkpoints_esb 3)

- 각 모델은 트레이닝 데이터의 [0:30000] 라인 까지의 데이터를 사용했다. 

  24000 : 6000 의 비율로 학습/검증 데이터셋을 나누고 훈련/검증 데이터는 shuffle 되어 학습되었다.

1. HardVoting
   - [re_nmt_Ensemble_HardVoting.ipynb](./re_nmt_Ensemble_HardVoting.ipynb)
2. SoftVoting
   - [re_nmt_Ensemble_SoftVoting.ipynb](./re_nmt_Ensemble_SoftVoting.ipynb)

➡️ 각 보팅 방법에 대한 정확도는 피피티에 설명 있음.

 

#### 🔎 Ensemble 을 이용한 NMT 정확도 확인(Model 5개,30000 line)

- Model 개수 : 5개





#### 🔎 Ensemble 을 이용한 NMT 정확도 확인(Model 5개, 60000 line)

- Model 개수 : 5개
- 트레이닝 데이터 : [spa-eng/spa_for_esb.txt](./spa-eng/spa_for_esb.txt)
- [Training Checkpoints](./5 Models Checkpoints_60000) 

1. HareVoting
   - 파일 : [nmt_Ensemble_HCompare.ipynb](./nmt_Ensemble_HCompare.ipynb)
   
2. SoftVoting
   - 파일 : [nmt_Ensemble_SCompare.ipynb](./nmt_Ensemble_SCompare.ipynb)
   
3. 5개 single model VS 5 Model Ensemble 정확도 비교

   - 파일 : [Accuracy_Compare.xlsx](./Accuracy_Compare.xlsx)

     





#### 🔎 Ensemble 을 이용한 NMT 정확도 확인 & Confidence 고려 (Model 5개, 60000 line)

- 고려할 것
  - 앙상블에 사용되는 단일 모델의 성능에 따라 더 좋은 결과를 앙상블 하는 것을 고려해보기 → "Confidence 고려"
  - 일단은 Soft Voting 방식이 Confidence 를 고려한다고 생각





#### 🔎 Super Model(5배 학습한 단일모델) & Ensemble 정확도 비교

- Super Model ~~: 데이터 5배 학습한 단일 모델~~

  - ~~파일(Test) : nmt_SuperModel.ipynb~~
  - ~~파일(Training) : nmt_SuperModel_Training.ipynb~~

  💥 데이터 부족💥 

  

- ~~Ensemble Model : time-step 마다 5 sigle model 의 앙상블을 적용한 모델~~

- Super Model : 5배 학습한 단일 모델 👉 Epoch 을 5배로 학습

  - 파일(Test) : [nmt_SuperModel.ipynb](./Super_Model/nmt_SuperModel.ipynb)
  - 파일(Training) : [nmt_SuperModel_Training.ipynb](./Super_Model/nmt_SuperModel_Training.ipynb)
  - Training Checkpoint : [super_checkpoint](./Super_Model/super_checkpoint)
  - 학습 검증용 테스트 데이터셋 : [test_data.txt](./Super_Model/test_data.txt)

- [Ensemble Model(Soft Voting)](./nmt_Ensemble_SCompare.ipynb) : time-step 마다 5 single model 의 앙상블을 적용한 모델

- Super Model VS Ensemble 정확도 비교

  - [Accuracy_Compare.xlsx](./Accuracy_Compare.xlsx)
  - [SuperModel_Result.xlsx](./Super_Model/SuperModel_Result.xlsx)

- 정확도 비교

  일단은 30개 정도의 문장으로 manualy 비교를 진행하였다. 

  - 정확하게 맞춘 비율 : Reference 문장과 정확히 일치

  - 정확 + 해석 동일 : Reference 문장과 정확히 일치 + Reference 문장과 정확히 일치하진 않지만 뜻은 동일
  - 정확 + 해석 + 약간 : Reference 문장과 정확히 일치 + Reference 문장과 정확히 일치하진 않지만 뜻은 동일 + 해석을 약간 다르게 함

  |                    | Super Model | Ensemble (Soft  Voting) | Ensemble-Model1 | Ensemble-Model2 | Ensemble-Model3 | Ensemble-Model4 | Ensemble-Model5 |
  | ------------------ | ----------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
  | 정확하게 맞춘 비율 | **0.24**    | 0.2                     | 0.16            | 0.22            | 0.16            | 0.12            | 0.16            |
  | 정확 + 해석 동일   | 0.36        | **0.4**                 | 0.18            | 0.36            | 0.28            | 0.32            | 0.36            |
  | 정확 + 해석 + 약간 | **0.46**    | **0.46**                | 0.36            | **0.46**        | 0.38            | 0.44            | 0.44            |

  - manualy test 를 진행했을 때 앙상블이 단일모델/Super 모델 보다 아주 조금 좋아보인다. 현재 test data 가 작으므로 좀 더 늘려서 더 정확한 정확도 비교가 필요하다.

- 해결방안

  - 기계 번역의 성능측정에 주로 **BLEU** 가 사용된다.

  - Python Package 의 nltk 안에 있는 BLEU 측정 방법을 사용한다.

    

#### 🔎 Ensemble VS Ensemble 에 사용된 단일 모델의 성능 비교 - BLEU 측정

- 각 모델 별 BLEU Score 측정 표 : [BLEU_Score.xlsx](./BLEU/BLEU_Score.xlsx)
  - [Ensemble Model](./BLEU/nmt_Ensemble_SCompare(for BLEU).ipynb)
  - [Model 1 ~ Model 5 (Ensemble 개별 모델)](./BLEU/re_nmt_Ensemble_Models(for BLEU).ipynb)
  - [Test Data](./BLEU/test_data.txt) : 학습하지 않은 1000개의 spa-eng 데이터
  - [Training Data](./BLEU/spa-eng(for BLEU).txt)
  - [Training Checkpoint](./5 Models Checkpoints_60000) : 트레이닝 데이터 1~60000 line
  - [BLEU 계산 코드](./BLEU/Calculate_BLEU.ipynb)

  |           | n-gram      | Ensemble (Soft  Voting) | Ensemble-Model1 | Ensemble-Model2 | Ensemble-Model3 | Ensemble-Model4 | Ensemble-Model5 |
  | --------- | ----------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
  | BLEU 평균 | 1-gram      | 0.036147909             | 0.033892129     | 0.036007992     | 0.035975691     | 0.034695139     | 0.034659121     |
  | 2-gram    | 0.000452381 | 0.000571429             | 0.0005          | 0.000452381     | 0.000666667     | 0.000571429     |                 |
  | 3-gram    | 0.000166667 | 0.000166667             | 0.0002          | 0.000166667     | 0.0002          | 0.000166667     |                 |
  | 4-gram    | 0           | 0                       | 0               | 0               | 0               | 0               |                 |

- 문제 
  - BLEU 의 정확한 측정을 위해 Reference(ex.사람이 번역한 문장) 문장이 많아야한다. 
  - 현재 코드에서 사용한 Test Dataset 은 학습 시 사용하지 않은 데이터셋으로, Candidate (Source 문장) 에 해당하는 Reference 문장이 1개 밖에 없다.
  - 따라서 BLEU 점수가 매우 낮게 나와 성능 비교를 판단하기가 어려운 문제가 있다.

- 해결방안

  - Reference 문장이 많이 있는 데이터를 찾는다. → 지금까지 찾아봤을 땐 없음
  - 대량의 번역 테스트를 할 수 있는 방법을 찾아본다.
    - [WMT 학회](http://statmt.org/wmt21/index.html)에서 진행하는 Shared Task 의 Competiton 을 통해 테스트를 진행해 볼 수 있을 것 같다.
    - 여러가지 Shared Task 중에 기계번역 관련 Competition 에서 Submission 을 시도해 볼 수 있지만, 현재 Submission 이 마감되어서 다음에 시도해보면 좋을 것 같다.
  - 기계 번역 성능을 측정하는 다른 평가 방법을 찾아본다. → 일단 찾아본 것 중에 Python Package 에 있는 Tranlation Quality Estimator 사용해본다.

  

#### 🔎 Ensemble VS Ensemble 에 사용된 단일 모델의 성능 비교 - QE(in Python Package) 측정

- 각 모델 별 QE Score 측정 표 : [BLEU_Score.xlsx ](./BLEU/BLEU_Score.xlsx) 안에 QE Sheet

  - [Ensemble Model](./BLEU/nmt_Ensemble_SCompare(for\BLEU).ipynb)

  - [Model 1 ~ Model 5 (Ensemble 개별 모델)](./BLEU/re_nmt_Ensemble_Models(for BLEU).ipynb)

  - [Test Data](./BLEU/test_data.txt) : 학습하지 않은 1000개의 spa-eng 데이터

  - [Training Data](./BLEU/spa-eng(for\BLEU).txt)

  - [Training Checkpoint](./5 Models Checkpoints_60000) : 트레이닝 데이터 1~60000 line

  - [QE 계산 코드](./BLEU/Calculate_QE.ipynb)

    |                                               | Ensemble (Soft  Voting) | Ensemble-Model1 | Ensemble-Model2 | Ensemble-Model3 | Ensemble-Model4 | Ensemble-Model5 |
    | --------------------------------------------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
    | Translation  Quality Estimator (QE)      평균 | **0.748706**            | 0.680984        | 0.710045        | 0.698374        | 0.701656        | 0.705305        |
    |                                               |                         

- 문제 

  - 성능 측정에 사용되는 Python Package 의 TQE 가 신뢰할 수 있는 것인지 잘 모른다.
  - 일단 manually 비교를 해봤을 땐 어느정도 비슷하면 숫자가 높게 나오는 것으로 보인다.
  - 이 방식으로 성능 비교를 해봤을 때 Ensemble 의 성능이 아주 조금 높게 나오는 것을 확인할 수 있다. (~~교수님께서 저 정도는 유의미한 향상이라고 할 수 있을 것 같다고 하심~~)

- 해결방안
  - 더 좋은 성능 측정 방법이 있는지 계속 찾아봐야겠다.

#### **🙌 Natural Language Translation 에서 단일 모델 모다 Ensemble 모델의 성능이 조금 더 좋다는 것을 확인할 수 있었다. 🙌**



→ 이어지는 다음 스터디 및 연구

#### 📝 Ensemble 서바이벌 

- [Repository](https://github.com/aaajeong/Survival-Ensemble)


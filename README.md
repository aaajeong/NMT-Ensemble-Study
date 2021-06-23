# 🎓Attention-Ensemble-Translation🎓
- [RNN Translation](https://github.com/aaajeong/RNN-Translation) - Tensorflow Attention 기계번역 Ensemble  연구 이어서



#### 📝 연구 주제 : Natural Language Translation 에서 Ensemble 을 사용하면 성능이 좋아질까?

---

1. **nmt_with_attention.ipynb**

   - [코드](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attention.ipynb)
   - [Training DataSet](https://drive.google.com/drive/u/0/folders/1pRSxMkYVS2ZyDO9P43VxzWfziiqZxT4G) : spa.txt - [0:30000]
   - Training Checkpoint: [training_checkpoints](https://drive.google.com/drive/u/0/folders/1WoHsFnNmx6xagcnkrl7bOM1SNw7qLWmw)

   - [**nmt_with_attention_Test.ipynb**](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attetion_Test.ipynb)

2. **nmt_with_attetntion_Ensemble_HardVoting.ipynb**

   - [코드](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attetion_Ensemble_HardVoting.ipynb)
   - [Training DataSet](https://drive.google.com/drive/u/0/folders/1pRSxMkYVS2ZyDO9P43VxzWfziiqZxT4G) : spa.txt - [0:30000], [30000:60000], [60000:90000]
   - Training Checkpoint : [training_checkpoints](https://drive.google.com/drive/u/0/folders/1WoHsFnNmx6xagcnkrl7bOM1SNw7qLWmw), [training_checkpoints_2](https://drive.google.com/drive/u/0/folders/1Vx4OU_4Agiq36OH9LKGTfzqSmshbW9Da), [training_checkpoints_3](https://drive.google.com/drive/u/0/folders/1-krO5p1I3aV5J4HKlOjnsSLXmgOG3PIf)

   **💥 문제 발견 💥**

3. **nmt_with_attention_Ensemble_SoftVoting.ipynb**

   - [코드](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attetion_Ensemble_SoftVoting.ipynb)
   - [Training DataSet](https://drive.google.com/drive/u/0/folders/1pRSxMkYVS2ZyDO9P43VxzWfziiqZxT4G) : spa.txt - [0:30000], [30000:60000], [60000:90000]
   - Training Checkpoint : [training_checkpoints](https://drive.google.com/drive/u/0/folders/1WoHsFnNmx6xagcnkrl7bOM1SNw7qLWmw), [training_checkpoints_2](https://drive.google.com/drive/u/0/folders/1Vx4OU_4Agiq36OH9LKGTfzqSmshbW9Da), [training_checkpoints_3](https://drive.google.com/drive/u/0/folders/1-krO5p1I3aV5J4HKlOjnsSLXmgOG3PIf)

   **💥 문제 발견 💥**

4. **nmt_with_attention_randomly_data.ipynb**

   - [코드](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attention_randomly_data.ipynb)
   - [Training DataSet](https://drive.google.com/drive/u/0/folders/1pRSxMkYVS2ZyDO9P43VxzWfziiqZxT4G) : shuffle_spa-eng.txt [0:30000]
   - Training Checkpoint : [training_checkpoint_randomly data](https://drive.google.com/drive/u/0/folders/1-01hD59JDPZWdH1bZQXHspwKSVNti4B9)

5. **nmt_with_attention_Ensemble_SoftVoting(2).ipynb**

   - [코드](https://github.com/aaajeong/Attention-Ensemble-Translation/blob/main/nmt_with_attetion_Ensemble_SoftVoting(2).ipynb)
   - [Training DataSet](https://drive.google.com/drive/u/0/folders/1pRSxMkYVS2ZyDO9P43VxzWfziiqZxT4G) : shuffle_spa-eng.txt [0:30000]. shuffle_spa-eng2.txt [0:30000], shuffle_spa-en3g.txt [0:30000]
   - Training Checkpoint : [training_checkpoint_randomly data](https://drive.google.com/drive/u/0/folders/1-01hD59JDPZWdH1bZQXHspwKSVNti4B9), [training_checkpoint_randomly data2](https://drive.google.com/drive/u/0/folders/1-2qkWld7dhOPRPnS8HzqSYZuuSrC0j_f), [training_checkpoint_randomly data3](https://drive.google.com/drive/u/0/folders/1-jRAs-1mgZDhEfZ4iMXIYSZTJETYSaBK)

   **💥 문제 발견 💥**

---

#### 💥문제 발견 설명💥

	- [앙상블 보팅 과정에서 모델 동작 확인](https://github.com/aaajeong/Attention-Ensemble-Translation/commit/5799477fe58b9194502bbcf8ac0a4f5100d00fdc)
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

- 트레이닝 데이터 : spa-eng/spa_for_esb.txt

- Training Checkpoint : 각 3개의 모델에 대한 파일

  - Model 1 : training_checkpoints_esb
  - Model 2 : training_checkpoints_esb 2
  - Model 3 : training_checkpoints_esb 3

- 각 모델은 트레이닝 데이터의 [0:30000] 라인 까지의 데이터를 사용했다. 

  24000 : 6000 의 비율로 학습/검증 데이터셋을 나누고 훈련/검증 데이터는 shuffle 되어 학습되었다.

1. HardVoting
   - re_nmt_Ensemble_HardVoting.ipynb
2. SoftVoting
   - re_nmt_Ensemble_SoftVoting.ipynb

➡️ 각 보팅 방법에 대한 정확도는 피피티에 설명 있음.

 

#### 🔎 Ensemble 을 이용한 NMT 정확도 확인(Model 5개)

- Model 개수 : 5개





#### 🔎 Ensemble 을 이용한 NMT 정확도 확인(Model 5개, 60000 line)

- Model 개수 : 5개
- 트레이닝 데이터 : 
- Training Checkpoints : 

1. HareVoting
   - 파일 : nmt_Ensemble_HCompare.ipynb
   
2. SoftVoting
   - 파일 : nmt_Ensemble_SCompare.ipynb
   
3. 5개 single model VS 5 Model Ensemble 정확도 비교

   - 파일 : Accuracy_Compare.xlsx

     





#### 🔎 Ensemble 을 이용한 NMT 정확도 확인 & Confidence 고려 (Model 5개, 60000 line)

- Model 개수 : 5개







#### 🔎 Super Model(5배 학습한 단일모델) & Ensemble 정확도 비교

- Super Model : 데이터 5배 학습한 단일 모델
  - 파일 : 
- Ensemble Model : time-step 마다 5 sigle model 의 앙상블을 적용한 모델




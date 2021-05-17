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
     - 새로운 파일 : re-nmt_Ensemble_HardVoting.ipynb , re-nmt_with_Ensemble_SoftVoting.ipynb




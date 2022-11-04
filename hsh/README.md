# Counter : Density map
농산품 Count를 위한 Density map 개선
</br></br>

## Purpose
Transfer learning 등의 추가 학습 없이 Model 개선 방법 학습 및 연구  
- scaling 과정에서의 annotation box modify  
- CNN 내 다른 layer의 feature extraction
- backbone model 교체
</br>

## Model  
FamNet
</br>

## Data Sample
기존 모델로 test시 상대적으로 높은 error가 발생하는 딸기 이미지 (Dataset: FSC147)
</br>

## Result

![example](https://github.com/Farmer-from-Space/midlevel_project/blob/main/hsh/img/counter%20result.png)
</br>

## Conclusion and Discussion
- 다양한 방법론의 CV model 학습
- Model의 Process와 깊은 구조에 대한 이해
- Dataset과 training의 중요성
- Model Generalization의 어려움
</br>

### References
[Learning To Count Everything](https://github.com/cvlab-stonybrook/LearningToCountEverything):
Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai</br>
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

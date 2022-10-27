## 개인 선정 주제 : 소비자의 시선 예측 모델을 사용한 Probability map 
- Input : 2개 이상의 상품을 포함한 이미지
- Output : pixel 당 fixation probability 를 나타낸 density map 

## 사용 데이터 샘플
- 선정 이유
사진 안에 여러 개의 상품이 포함되어 있어 다양한 상품 중 어떤 것에 시선이 먼저 가는지 실험하기 위한 데이터셋으로 적합함

## 전제 및 전처리
전제 : 사람의 인식과 관련된 이미지의 특징을 조정함으로써 모델의 성능을 향상시킬 수 있다 <br>
전처리 항목  <br>
- L*a*b color space <br>
- gamma adjustment

## Results

## Further development

- Trainable channel인 readout & finalization 의 재학습 <br>
(단, 신뢰성 있는 fixation labeling이 완료된 데이터셋 필요)
- 움직이는 영상의 실시간 처리

## Application

- 매대 사진에서 시선이 먼저 가는 위치 혹은 상품 확인하여 배치에 활용
- 온라인 홈페이지 이미지에 적용하여 광고 배치 및 효과 확인

## References

<details>
<summary>Understanding Low- and High-Level Contributions to Fixation Prediction</summary>
<div markdown="1">

M. Kümmerer, T. S. A. Wallis, L. A. Gatys and M. Bethge, "Understanding Low- and High-Level Contributions to Fixation Prediction," 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 4799-4808, doi: 10.1109/ICCV.2017.513.

</div>
</details>

<details>
<summary>DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling</summary>
<div markdown="1">
Akis Linardos, Matthias Kümmerer, Ori Press, Matthias Bethge; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 12919-12928
</div>
</details>

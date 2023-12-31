## 패션 스타일에 따른 쇼핑몰 및 의류 추천 시스템

### Overview
- 옷의 스타일 분류를 위해 resnet50 모델 사용
- 사진에서 객체들을 검출해 bounding box를 씌우기 위해 Mask R-CNN 모델 사용
- 추천을 위해 사진 간 유사도 도출을 위해 VGG16 모델 사용

### 파일 구성
#### 데이터셋
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=75 
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=51
- 직접 인스타에서 크롤링한 데이터

#### 모델
- rec_model 2.pkl : resnet50, Mask R-CNN, VGG16 모델에 대한 내용 포함
- insta_1.ipynb : 인스타 크롤링에 대한 내용 포함
- hello.py : flask에서 구동하기 위한 python 파일
- static, templates : flask 구동을 위한 html, css, img 파일 포함
- recommend_model.ipynb : model 폴더의 pickle 파일의 원본

### 구동 사진
![image](https://github.com/fkrdnjs/kdata_fashion/assets/68600918/b977cb3d-20a8-41a2-9c1d-bcb89c06de54)
![image](https://github.com/fkrdnjs/kdata_fashion/assets/68600918/510d1831-1eee-4a12-8f7d-308d880726b2)
![image](https://github.com/fkrdnjs/kdata_fashion/assets/68600918/10a732d4-21af-4115-bceb-0554dc20d8fb)

<h1>Car_model-classification</h1>

**코드스테이츠 기업협업 프로젝트로 '넥스트랩' 회사와 함께 진행했음을 알립니다.**<br>
참여자: 황산하, 조미래, 박정기, 김홍균, 이재웅, 신정태<br>

데이터 용량에 문제로 인해 사용한 코드와 툴, 아이디어만 깃헙 업로드하였습니다.<br>
Yolov4에 대한 원본 source는 https://github.com/kairess/tensorflow-yolov4-tflite 입니다. 수정한 파일만 업로드하였습니다.<br>



<h3>프로젝트 가설</h3>  
사람이 차를 보고 모델과 연식을 구분해내는 기준은 차의 그릴, 헤드라이트, 범퍼 등을 연상할 수 있습니다.<br>
CNN기반의 efficientNet model또한 차량 이미지속 그 부분을 보고 판단할 지, 오분류를 한다면 어떤 부분이 문제인지 검증할 필요를 느꼈습니다.

<h3>프로젝트 목표</h3>    
1) 차량 이미지 데이터로 학습한 EfficientNet model을 통해 영상 또는 이미지속 차량을 model명, 연식 등을 구분<br>
2) 시각적 정보를 처리하는 과정을 본따만든 CNN기법의 검증       



<h3>데이터형태</h3>   

<p align='center'><img src="https://user-images.githubusercontent.com/76782896/133728623-8f55f6b6-12eb-444d-bcd0-b5e4b28efa4f.png" width="30%" height="30%"/></p>
<p align='center'><img src="https://user-images.githubusercontent.com/76782896/133728159-9a2afc2d-9422-4b26-bc32-381712eb71bf.png" width="50%" height="50%"/></p>

<h3>Classification 종류</h3>   
개체의 종류가 다른 Image classification 문제와는 달리, 종류가 같지만 미묘하게 조금씩 다른 Fine-grained Image classification 문제로 인식하였습니다.<br>

<p align='center'><img src="https://user-images.githubusercontent.com/76782896/133729598-56703693-1246-4816-bac4-ac77eeff28fa.png" width="50%" height="50%"/></p>

<h3>Image Augmentation</h3>   
각 분류 문제마다 알맞은 augmentation 기법을 사용할 필요를 인식하였습니다.<br>
<br>

   
A) 기존 증강기법과 도로상황에서 발생할 수 있는 사물에 의해 가려질 수 있는 상황을 가정하여 증강한 이미지
<p align='center'><img src="https://user-images.githubusercontent.com/76782896/133730231-103ff244-5036-4937-913f-b27c8699461c.png" width="20%" height="20%"/></p>   

B) 조금 더 헤드라이트, 범퍼, 그릴의 형태가 잘 드러낼 수 있도록 증강기법을 활용하여 증강한 이미지   
<p align='center'><img src="https://user-images.githubusercontent.com/76782896/133730456-678696da-cf22-4ae3-8aa8-340eae08e1c3.png" width="40%" height="40%"/></p>   


<h3>활용한 Deep Learning model</h3>   
EfficientNet V2를 사용하였으며, 여러 version중 B0, B1 version을 활용하였습니다. <br>

<p align='center'>
  <img src="https://user-images.githubusercontent.com/76782896/133730699-d384a1a1-33bd-4d3f-8a80-c21581c5d4a4.png" width="400px" height="400px"/>
  <img src="https://user-images.githubusercontent.com/76782896/133730705-b86a4e85-57a8-470f-877a-4ea53e2cea9b.png" width="300px" height="400px"/>
</p> 

<h3>모델 평가</h3>
확연히 학습한 데이터의 수가 많을수록 학습이 잘되고, 일반화도 잘되는 모습을 볼 수 있었고, Fine-tunning을 진행하였을때 (설명 보완할 필요가 있음)
부분 파라미터를 홀딩시키고 부분만 학습시키는 경우가 더 좋은 성능을 나타냈습니다. <br>

평가지표는 val_loss와 val_accuracy를 활용하였다. <br>

<p align='center'>
  <img src="https://user-images.githubusercontent.com/76782896/133733135-6c0f9230-538a-49ee-835a-ce81e8160e01.png" width="400px" height="400px"/>
  <img src="https://user-images.githubusercontent.com/76782896/133733138-5f3b69c9-f770-4a9c-9a8b-75202f029449.png" width="300px" height="400px"/>
</p> 


<h3>목표1에 대한 결과</h3>
<p align='center'>
  <img src="https://user-images.githubusercontent.com/76782896/133732292-62b395a4-67b7-4e5a-8922-5d42d43edf32.png" width="300px" height="400px"/>
  <img src="https://user-images.githubusercontent.com/76782896/133732297-df696b96-d8c4-41f9-8fee-38aa8663c5c4.png" width="300px" height="400px"/>
  <img src="https://user-images.githubusercontent.com/76782896/133732540-1c418d75-f0fb-4736-b07b-f6b3d42417f3.png" width="300px" height="400px"/>
</p> 

왼쪽 사진: 자동차의 이미지가 정확하게 나온 사진, top3안에 정답모델이 있음을 확인할 수 있었다.<br>
가운데 사진: 앞부분만 나온 사진, 잘못된 분류<br>
오른쪽 사진: 앞부분이 가려진 사진, 낮은 확률로 예측이 진행되었고, top3안에 비슷한 모델을 말할 수 있다.

<h3>목표2에 대한 결과</h3>

CNN의 가중치가 어느 부분을 높게 평가하고 있는지 시각적을 확인하기 위해 히트맵을 작성하였습니다.<br>
가설 설정과 같이 CNN의 가중치 또한 차량의 범퍼, 그릴, 헤드라이트부분을 높게 부여한 것을 확인할 수 있었고, 뿐만아니라 차량 앞유리 shape, 옵션으로 달 수 있는 썬루프 등을 분류하는데 활용함을 알 수 있었습니다. <br>

<p align='center'>
  <img src="https://user-images.githubusercontent.com/76782896/133734030-71c0642b-aafe-4533-b597-f0cba5c1aedd.png" width="400px" height="400px"/>
  <img src="https://user-images.githubusercontent.com/76782896/133734237-fb6dc2f3-9f4b-4c84-bd2b-87c059b115a3.png" width="400px" height="400px"/>
</p> 

<h3>증강기법에 따른 테스트 결과</h3>

증강기법A을 통한 test accuracy : 74.04%<br>
증강기법B를 통한 test accuracy : 75.11%<br>

큰 차이는 없었으나 증강기법에 따라 차이가 있음을 확인할 수 있었고, 조금 더 Fine-grained classification에 맞는 증강기법을 더욱 추가하고 세분화한다면 확연한 차이를 보일 수 있지 않을까 생각합니다.

<h3>활용방안 및 향후 과제</h3>

차량 분석을 통해 터미널의 배기가스량을 예측하여 터미널 환기시스템에 적용할 수 있다.<br>
도로를 다니는 차량에 따라서 도로의 파손여부를 예측하여, 도로 보수공사 필요성에 분석결과를 적용할 수 있다. <br>
<br>
<br>
<br>
<br>

#######################################################################################<br>

**class_augmentation 파일**

원작자: 이재웅, 신정태, 김홍균<br>
수정자: 황산하, 박정기, 조미래<br>

Augmentation 기법 A: 이재웅(주도), 신정태, 김홍균<br>
Augmentation 기법 B: 조미래(주도), 황산하, 박정기<br>

**train data set과 validation data set split방법**
Util 파일 수정: 박정기 황산하<br>
Util_2_2 파일: 황산하, 박정기 작성<br>

#######################################################################################<br>
**EfficientNet_차종분류 ipynb 파일**

원작자: 황산하, 박정기, 조미래<br>

히트맵 코드 구현자: 박정기, 이재웅<br>

테스트 결과 폴더 생성 코드 협업자: 이재웅<br>

#######################################################################################<br>

**YOLOv4를 활용하여 자동차 detection 및 crop**

수정자: 황산하<br>

######################################################################################################<br>

**make_test_data.ipynb**

원작자: 박정기<br>




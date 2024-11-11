'''
* 역할부여
당신은 정말 쉽게 파이썬을 가르쳐 주는 교사입니다.
저는 파이썬 기초 문법 정도만 알고 있는 입무자 수준의 5학년 학생입니다.
지금부터 단계별로 질문하겠습니다.
각 질문에 맞춰서 답변해 주세요

* 패키지 소개 받기
사람 얼굴이 포함된 이미지에서 사람 얼굴을 모자이크 처리하는 파이썬 프로그램을 만들고 싶습니다.
이 프로그램을 만들때 사용할 수 있는 파이썬 패키지 3개를 장단점과 함께 소개해 주세요
이 패키지 중에서 가장 많은 사람들이 사용하고 초보자가 사용하기 쉬운 패키지를 선택해 주고 그 이유를 알려주세요.
코드는 필요없습니다.

* 패키지 함수 소개
opencv에서 가장 많이 사용되는 함수 10개를 알려주세요
각 함수의 이름과 간략한 설명, 쉽고 간단한 예시 코드를 보여주세요

* 패키지를 이용한 프로그램 만들기
사람 얼굴이 포함된 이미지에서 사람 얼굴을 모자이크 처리하는 파이썬 프로그램을 만들고 싶습니다.
opencv에서 가장 많이 사용되고 사용하기 쉬운 함수를 조합해서 최대한 간단한 코드를 작성해 주세요.
각 기능마다 주석이 달려있어야 하고 코드에는 오류가 없어야 합니다.
제가 잘 이해할 수 있도록 각 코드 라인마다 구체적으로 설명해 주세요

모자이크된 이미지를 저장하는 코드도 추가해줘

'''

import cv2  # OpenCV 라이브러리를 불러옵니다.

# 1. 이미지 읽기
image = cv2.imread('image.jpg')  # 'image.jpg' 파일에서 이미지를 읽어옵니다.
# 이미지를 그레이스케일로 변환하여 얼굴 인식 속도를 높입니다.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 얼굴 인식을 위한 분류기 로드
# OpenCV에서 제공하는 얼굴 검출용 Haar Cascade 분류기를 로드합니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. 얼굴 검출
# detectMultiScale 함수는 이미지에서 얼굴을 찾아 위치와 크기를 반환합니다.
# scaleFactor는 이미지 크기 조정을 위한 값, minNeighbors는 얼굴 검출의 정확도 조정 값입니다.
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# 4. 얼굴 영역에 모자이크 처리
for (x, y, w, h) in faces:  # 검출된 얼굴의 위치(x, y)와 크기(w, h)를 반복문으로 처리합니다.
    # 얼굴 부분만 잘라냅니다.
    face = image[y:y + h, x:x + w]

    # 모자이크 처리를 위해 얼굴 이미지를 작은 크기로 축소합니다.
    face_small = cv2.resize(face, (w // 10, h // 10))

    # 축소한 얼굴 이미지를 다시 원래 크기로 확대합니다.
    face_mosaic = cv2.resize(face_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 원본 이미지에서 얼굴 부분을 모자이크 이미지로 교체합니다.
    image[y:y + h, x:x + w] = face_mosaic

# 5. 모자이크된 이미지 저장
cv2.imwrite('mosaic_image.jpg', image)  # 모자이크된 이미지를 'mosaic_image.jpg'로 저장합니다.
print("모자이크된 이미지를 저장했습니다.")

# 6. 결과 이미지 출력
cv2.imshow('Mosaic Image', image)  # 모자이크 처리가 된 이미지를 화면에 표시합니다.
cv2.waitKey(0)  # 키 입력을 기다립니다.
cv2.destroyAllWindows()  # 모든 창을 닫아줍니다.

'''
* 상황 설명하기 (역할부여)
당신은 정말 쉽게 파이썬을 가르쳐 주는 교사입니다.
저는 파이썬 기초 문법 정도만 알고 있는 입문자 수준의 5학년 학생입니다.
지금부터 단계별로 질문하겠습니다.
각 질문에 맞춰서 답변해 주세요

* 패키지 선택하기
웹사이트 링크를 QR코드로 생성하는 파이썬 프로그램을 만들고 싶습니다.
이 프로그램을 만들때 사용할 수 있는 파이썬패키지 3개를 장단점과 함께 소개해 주세요
이 패키지 중에서 가장 많은 사람들이 사용하고 초보자가 사용하기 쉬운 패키지를 선택해 주고 그 이유를 알려주세요.
코드는 필요없습니다.

* 패키지 파악하기
qrcode에서 가장 많이 사용되는 함수 10개를 알려주세요
각 함수의 이름과 간략한 설명, 쉽고 간단한 예시 코드를 보여주세요

* 파이썬 코드 요청하기
웹사이트 링크를 QR코드로 생성하는 파이썬 프로그램을 만들고 싶습니다.
qrcode에서 가장 많이 사용되고 사용하기 쉬운 함수를 조합해서 최대한 간단한 코드를 작성해 주세요.
각 기능마다 주석이 달려있어야 하고 코드에는 오류가 없어야 합니다.
제가 잘 이해할 수 있도록 각 코드 라인마다 구체적으로 설명해 주세요

* 수정하면서 완성하기
QR코드의 색깔을 초록색으로 바꾸도록 프로그램을 수정해 주세요
제가 잘 이해할 수 있도록 각 라인별로 구체적으로 설명해 주세요
'''

# qrcode 라이브러리를 불러옵니다.
import qrcode

# QR코드에 포함할 웹사이트 링크를 설정합니다.
url = "https://www.sdu.ac.kr"  # 이 부분에 원하는 웹사이트 링크를 입력하세요.

# 웹사이트 링크를 사용해 QR코드를 생성합니다.
img = qrcode.make(url)  # qrcode.make()는 간단히 QR코드를 생성해주는 함수입니다.

# 생성된 QR코드를 화면에 표시합니다.
img.show()  # img.show()는 이미지를 열어 화면에 보여줍니다.

# QR코드를 이미지 파일로 저장합니다.
img.save("qrcode_example.png")  # img.save()는 QR코드를 이미지 파일로 저장합니다.

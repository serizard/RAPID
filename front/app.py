import streamlit as st
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
from PIL import Image

from webcam import record_screen  # webcam.py에서 record_screen 함수 불러오기
import openai
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
from pathlib import Path

# 실행 시
# streamlit run "c:\Users\nyny0\OneDrive\문서\3학년 2학기\캡스톤설계프로젝트\project\MMATD\backend\front\app.py"

if 'page' not in st.session_state:
    st.session_state.page = "start"  # 초기 화면 설정
    st.rerun()

if 'recording_complete' not in st.session_state:
    st.session_state.recording_complete = False  # 녹화 완료 상태 초기화
    st.rerun()

import base64
from pathlib import Path

def get_image_base64(image_path):
    """이미지 파일을 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def story_screen():
    st.markdown("<h2 style='text-align: center;'>Cinderella Story 📖</h2>", unsafe_allow_html=True)
    
    # 페이지 상태 초기화
    if 'story_page' not in st.session_state:
        st.session_state.story_page = 0
    
    # 기본 경로 설정
    base_path = Path(r"C:\Users\injun\OneDrive\대학교\3-2\캡스톤디자인\MMATD\Cinderella")
    
    # 스토리 페이지 데이터 - 한글 버전
    story_pages = [
        {
            "image_path": base_path / "1.png",
            "caption": "Once upon a time, in a little village, there lived a sweet girl named Cinderella. All the animals adored her and loved her dearly."
        },
        {
            "image_path": base_path / "2.png",
            "caption": "Cinderella lived with her stepmother and two stepsisters named Anastasia and Drizella. They made Cinderella clean, sew, and cook all day long."
        },
        {
            "image_path": base_path / "3.png",
            "caption": "Cinderella’s stepmother was jealous of her beauty and treated her with coldness and cruelty. Yet, kind-hearted Cinderella tried her best to earn their love."
        },
        {
            "image_path": base_path / "4.png",
            "caption": "One day, a special invitation arrived for a grand ball at the royal palace. The king hoped that the prince would find a bride, so all the unmarried young ladies in the kingdom were invited."
        },
         {
            "image_path": base_path / "5.png",
            "caption": "Cinderella was overjoyed at the thought of going to the ball. She found her mother’s old dress in the attic and decided to make it beautiful so she could wear it to the ball."
        },
         {
            "image_path": base_path / "6.png",
            "caption": "Cinderella’s stepmother didn’t want her to go to the ball. So, she kept giving Cinderella more and more chores—tasks that would take her all evening to finish."
        },
         {
            "image_path": base_path / "7.png",
            "caption": "While Cinderella worked, her animal friends fixed up her dress. They added pretty ribbons and beads that her stepsisters had thrown away, turning it into a beautiful gown."
         },
         {
            "image_path": base_path / "8.png",
            "caption": "Cinderella was overjoyed when she saw the dress her animal friends had fixed up for her. Now, she could go to the ball too! She thanked her little friends with all her heart."
        },
         {
            "image_path": base_path / "9.png",
            "caption": "But when her stepsisters saw the ribbons and beads on Cinderella’s dress, they were furious. They grabbed at the beads and ribbons, pulling them off until the dress was ruined."
        },
         {
            "image_path": base_path / "10.png",
            "caption": "Heartbroken, Cinderella ran into the garden, tears streaming down her face. But suddenly, her fairy godmother appeared! With a wave of her magic wand, she turned a pumpkin into a magnificent carriage."
        },
         {
            "image_path": base_path / "11.png",
            "caption": "“Bibbidi-Bobbidi-Boo!”\nIn an instant, Cinderella was transformed, dressed in a beautiful gown with sparkling glass slippers. But her fairy godmother warned her that the magic would fade at midnight."
        },
         {
            "image_path": base_path / "12.png",
            "caption": "At the ball, the prince saw Cinderella and couldn’t take his eyes off her beauty. As the music began to play, the prince started to dance with the lovely Cinderella."
        },
         {
            "image_path": base_path / "13.png",
            "caption": "But as the clock struck midnight, Cinderella’s magical evening came to an end. She quickly left the ballroom with only a hurried goodbye, leaving behind a single glass slipper."
        },
         {
            "image_path": base_path / "14.png",
            "caption": "The prince sent out his servants to find the girl whose foot would fit the glass slipper. Despite the stepmother’s attempts to interfere, it was finally revealed that Cinderella was the true owner of the glass slipper!"
        },
          {
            "image_path": base_path / "15.png",
            "caption": "Cinderella and the prince soon had their wedding, and everyone celebrated their happiness together."
        },
          {
            "image_path": base_path / "16.png",
            "caption": "Amid the blessings of everyone around them, the prince and Cinderella lived happily ever after."
        }
    ]
    
    # 현재 페이지 표시
    current_page = story_pages[st.session_state.story_page]
    
    # 스타일 정의
    st.markdown("""
        <style>
        .storybook-container {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 800px;
        }
        .story-image {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .story-caption {
            font-family: 'Nanum Gothic', sans-serif;
            font-size: 18px;
            line-height: 1.8;
            text-align: center;
            padding: 20px;
            color: #333;
            white-space: pre-line;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 10px 0;
        }
        .page-number {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin: 10px 0;
        }
        .nav-button {
            background-color: #4e73df;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .nav-button:hover {
            background-color: #2e59d9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 이미지와 캡션 표시
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        # 이미지를 base64로 인코딩
        image_base64 = get_image_base64(current_page["image_path"])
        
        st.markdown(f"""
            <div class="storybook-container">
                <img src="data:image/png;base64,{image_base64}" class="story-image">
                <div class="story-caption">{current_page["caption"]}</div>
                <div class="page-number">Page {st.session_state.story_page + 1} of {len(story_pages)}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 네비게이션 버튼
        col_prev, col_next = st.columns([1, 1])
        
        with col_prev:
            if st.session_state.story_page > 0:
                if st.button("◀ Previous Page", use_container_width=True):
                    st.session_state.story_page -= 1
                    st.rerun()
        
        with col_next:
            if st.session_state.story_page < len(story_pages) - 1:
                if st.button("Next Page ▶", use_container_width=True):
                    st.session_state.story_page += 1
                    st.rerun()
        
        # 돌아가기 버튼
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to the test", type="primary", use_container_width=True):
            st.session_state.page = "instruction"
            st.rerun()

# 시작 화면
def start_screen():
    st.markdown("<h2 style='text-align: center;'>Aphasia Diagnosis Test 🗣️</h2>", unsafe_allow_html=True)
    if st.button("Start test"):
        st.session_state.page = "instruction"
        st.rerun()

# 테스트 안내 화면
def instruction_screen():
    st.markdown("<h2 style='text-align: center;'>Test Guidance 📝</h2>", unsafe_allow_html=True)
    st.write("This test involves recording a video for aphasia diagnosis. When you're ready, please press the 'Continue Test' button to begin recording.")
    st.write("If you want to read the story of Cinderella before the test, please click the 'Review Cinderella Story' button")

    if st.button("Review Cinderella Story"):
        st.session_state.page = "story"
        st.rerun()
    
    if st.button("Continue Test"):
        st.session_state.page = "information" # 새로운 사용자 정보 화면으로 이동
        st.rerun()

# 사용자 정보 입력 화면
def information_screen():
    st.markdown("<h2 style='text-align: center;'>User Information</h2>", unsafe_allow_html=True)

    # 사용자 정보 입력 필드
    name = st.text_input("Name")

    # 출생년도 선택
    current_year = datetime.now().year
    birth_year = st.selectbox("Birth Year", list(range(1900, current_year + 1))[::-1])  # 1900년부터 현재 연도까지, 최신 연도가 위에 오도록 정렬

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # 사용자 정보를 세션 상태에 저장하고 녹화 화면으로 이동
    if st.button("Next"):
        st.session_state.user_info = {"name": name, "birth_year": birth_year, "gender": gender}
        st.session_state.page = "record"
        st.rerun()

# 결과 확인 화면
def result_screen():
    st.markdown("<h2 style='text-align: center;'>Complete test</h2>", unsafe_allow_html=True)
    st.write("The test has ended. Please click the button below to view the results.")
    if st.button("Show Results"):
        st.session_state.page = "pdf"
        st.rerun()


def pdf_screen():
    st.markdown("<h2 style='text-align: center;'>📄 Diagnostic Report</h2>", unsafe_allow_html=True)

    # 사용자 정보 불러오기
    user_info = st.session_state.get("user_info", {"name": "홍길동", "birth_year": 1977, "gender": "남성"})
    
    # 나이 계산
    current_year = datetime.now().year
    age = current_year - user_info["birth_year"]
    
    # openai key 설정
    openai.api_key = "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" 
    

    # 진단 보고서 내용

    pre_prompt = "한국어로 친절하게 대답해주세요. 실어증 Logit 값은 다음과 같습니다. ‘Wernicke: 0.2, Broca: 0.2, Anomic: 0.5, Control: 0.1’\n\n" # output으로 나온 logit 값을 여기다가! 
    prompt = f"""실어증 유형에 따른 진단 결과 보고서를 작성해주세요. 제목은 '<실어증 진단 결과 보고서>'로 하고, '제목: '은 빼 주세요. 각 섹션을 충실히 구성하고 아래 조건을 따르도록 작성해 주세요:
1.   환자 정보: '이름, 진단 날짜, 나이, 성별'이 포함되도록 작성해주세요. 각 항목은 다음과 같습니다.
- 이름: {user_info["name"]}
- 진단 날짜: {datetime.now().strftime('%Y년 %m월 %d일')}
- 나이: {age}세
- 성별: {user_info["gender"]}
2.   실어증 유형 진단: 위에서 주어진 실어증 유형별 로짓 값을 평가하고, 가장 높은 값을 가진 실어증을 주요 진단 유형으로 선택합니다. 
o   각 유형의 로짓 값을 바탕으로 진단하며, 로짓 값에 따른 해석을 명확하게 적어주세요.
3.   실어증 유형 설명: 위에서 진단한 실어증 유형에 대한 자세한 설명을 추가합니다.
4.   위험도 평가: 로짓 값을 바탕으로 위험도를 평가하여, 0.5 이상은 위험 수준, 0.3 이상은 의심 수준, 그 미만은 저위험으로 표시해주세요. 진단 유형의 위험도를 보고서에 명확히 기재합니다.
5.   세부 개선 사항: 정상인의 발화 task와 실어증 환자의 발화 task를 비교하여 차이점을 설명해 주세요. (예: 문법 오류 빈도, 의미 전달의 명확성, 이해 능력의 차이 등)
6.   권고 사항: 환자가 조기에 관리 및 치료할 수 있도록 권장사항을 포함합니다. 
o   전문 병원 상담, 언어 재활 프로그램의 참여, 그리고 가족과의 소통 강화를 권고합니다.
7.   결론: 진단의 중요성과 권장사항을 요약하여, 실어증 증상의 조기 치료와 관리의 중요성을 강조해 주세요.
다음과 같은 구조로 결과 보고서를 작성합니다:
•   환자 정보
•   실어증 유형 진단
•   실어증 유형 설명
•   위험도 평가
•   세부 개선 사항
•   권고 사항
•   결론
각 섹션을 아주 자세하게 작성하여 환자가 자신의 상태와 필요한 조치에 대해 이해할 수 있도록 해 주세요. 
주의: 대화형 답변이나 불필요한 서문, 마무리 문구는 포함하지 말고, 보고서 내용만 간결하게 작성해 주세요.""" # 이부분을 프롬프트 엔지니어링을 통해 수정 

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pre_prompt + prompt}
            ],
            max_tokens=3000,
            temperature=0.5
        )

        # 응답 텍스트 출력
        answer = response.choices[0].message.content.strip()
        st.write("", answer)
        
        # 그래프 생성 및 저장
        fig, ax = plt.subplots()
        labels = ['Wernicke', 'Broca', 'Anomic', 'Control']
        values = [0.2, 0.2, 0.5, 0.1]
        ax.bar(labels, values)
        ax.set_title("Logit Value by Type of Aphasia")
        ax.set_ylabel("Logit Value")

        # 그래프를 이미지로 저장
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # PDF 생성 함수
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 한글 폰트 추가
        pdf.add_font('NanumeGothic', '', r'C:\Users\injun\OneDrive\대학교\3-2\캡스톤디자인\MMATD\front\NanumGothic.ttf', uni=True)
        pdf.set_font('NanumeGothic', size=12)
        
        # 텍스트를 PDF에 추가 (multi_cell 사용)
        pdf.multi_cell(0, 10, txt=answer)
        
         # 그래프 이미지 삽입 시 위치 지정!!!!!!!!!
        pdf.image(img_buffer, x=10, y=pdf.get_y() + 10, w=100)
        
        # PDF를 바이트로 저장
        pdf_output = io.BytesIO()
        pdf.output(pdf_output, 'F')
        pdf_output.seek(0)  # Stream 시작 부분으로 이동
        
        # PDF 다운로드 버튼
        st.download_button(
            label="Download PDF",
            data=pdf_output,
            file_name="진단_결과_보고서.pdf",
            mime="application/pdf"
        )
        

    except Exception as e:
        st.error(f"error: {str(e)}")


# 페이지 전환 로직
if st.session_state.page == "start":
    start_screen()
elif st.session_state.page == "story":
    story_screen()
elif st.session_state.page == "instruction":
    instruction_screen()
elif st.session_state.page == "information":
    information_screen()  # 사용자 정보 입력 화면 추가
elif st.session_state.page == "record":
    record_screen()  # webcam.py에서 가져온 record_screen 함수 호출
elif st.session_state.page == "result":
    result_screen()
elif st.session_state.page == "pdf":
    pdf_screen()

import streamlit as st
import base64

# 🔐 로고 base64 인코딩
def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("assets/knu_logo.png")

# 🌐 페이지 설정
st.set_page_config(page_title="📘 경북대 챗봇", layout="centered")

# 🎨 전역 스타일
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Noto Sans KR', sans-serif;
    }
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 사이드바: 학사일정 + PDF + 링크
with st.sidebar:
    st.image("assets/knu_logo2.png", width=200)

    st.markdown("###  학사일정")
    st.markdown("""
    - 🗓️ 개강: **2025.09.01**  
    - 🖋️ 수강신청: **08.12 ~ 08.14**  
    - 📝 중간고사: **10.22 ~ 10.28**  
    - 💳 등록금 납부: **08.25 ~ 08.28**
    """)

    st.markdown("###  문서 다운로드")
    st.download_button(
        label="📄 등록금 납부 일정",
        data=open("data/2025학년도 2학기 등록금 납부 일정.pdf", "rb").read(),
        file_name="2025학년도_2학기_등록금_납부_일정.pdf",
        mime="application/pdf"
    )
    st.download_button(
        label="📄 강의평가",
        data=open("data/강의평가.pdf", "rb").read(),
        file_name="강의평가.pdf",
        mime="application/pdf"
    )
    st.download_button(
        label="📄 휴학 및 복학",
        data=open("data/경대 휴학,복학.pdf", "rb").read(),
        file_name="휴학및복학.pdf",
        mime="application/pdf"
    )

    st.markdown("###  바로가기 링크")
    st.markdown("- [ 경북대학교 홈페이지](https://www.knu.ac.kr)")
    st.markdown("- [ 종합정보시스템](https://appfn.knu.ac.kr/login.knu?agentId=4)")
    st.markdown("- [ 수강신청 페이지](https://sugang.knu.ac.kr)")
    st.markdown("- [ 시간표 조회 시스템](https://knuin.knu.ac.kr/public/stddm/lectPlnInqr.knu)")

# 🏫 상단 로고 + 타이틀
st.markdown(f"""
    <div style='display:flex; flex-direction:row; align-items:center; 
                justify-content:center; gap: 15px; margin-bottom: 32px;'>
        <img src="data:image/png;base64,{logo_base64}" style="height:80px;">
        <h2 style='margin: 30px 0 0 0; font-size: 45px; font-weight: 700; color:#212121;'>
            경북대학교 AI 도우미
        </h2>
    </div>
""", unsafe_allow_html=True)

# ✅ 대화 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! 📘 경북대 학사 도우미입니다. 무엇이든 물어보세요!"}
    ]

# ✅ 대화 렌더링
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "assistant":
        if i == 0:
            mascot_img = "assets/mascot_hello.png"
        elif i == len(st.session_state["messages"]) - 1:
            mascot_img = "assets/mascot_alarm.png"
        else:
            mascot_img = "assets/mascot.png"

        col1, col2 = st.columns([1, 8])
        with col1:
            st.image(mascot_img, width=130)
        with col2:
            st.markdown(f"""
                <div style='position:relative; background-color:#ffffff;
                            padding:15px 20px; border-radius:20px;
                            border: 1px solid #e0e0e0;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                            max-width: 90%; margin-bottom:15px;'>
                  {msg['content']}
                </div>
            """, unsafe_allow_html=True)

    elif msg["role"] == "user":
        st.markdown(f"""
            <div style='text-align:right; margin-bottom:15px;'>
                <div style='position:relative; display:inline-block; 
                            background-color:#b71c1c; color:white;
                            padding:15px 20px; border-radius:20px;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                            max-width: 85%;'>
                  {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)


# ✅ 자주 묻는 질문 버튼
frequent_questions = [
    "휴학은 어떻게 하나요?",
    "복학 신청은 어디서 하나요?",
    "수강신청 일정은 언제인가요?",
    "성적 열람은 어디서 하나요?",
    "학생증 발급은 어떻게 하나요?"
]
cols = st.columns(len(frequent_questions))
for idx, q in enumerate(frequent_questions):
    if cols[idx].button(q):
        st.session_state["messages"].append({"role": "user", "content": q})
        st.rerun()

# ✅ 사용자 입력
if user_input := st.chat_input("질문을 입력하세요 (예: 휴학 신청은 어떻게 하나요?)"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.rerun()

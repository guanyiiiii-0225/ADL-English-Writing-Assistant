import streamlit as st

from grammar_checker.inter import generate_html
from grammar_checker.extract2 import extract_from_html

def image_to_text(image):
    # convert the image to text
    image_description = "This is an image."
    return image_description

def generate_example(question):
    # wait 1 seconds
    import time
    time.sleep(1)
    # generate the example answer
    example = "I agree with the trend of universities encouraging professors to teach professional courses in English. This approach has several benefits. Firstly, it exposes students to a broader range of academic literature and research, as English is the lingua franca of higher education. This helps students to develop a more nuanced understanding of their field and stay up-to-date with the latest developments. Secondly, teaching in English promotes a more diverse and inclusive learning environment, as students from different countries and backgrounds can participate and engage with the course materials more easily.\n\nIf I were to enroll in a future university where the required courses are taught in English, I would embrace this opportunity to improve my language skills and broaden my academic horizons. I would take advantage of resources such as language support services and peer mentorship programs to help me succeed in these courses. Additionally, I would actively seek out opportunities to engage with course materials and participate in class discussions, using my knowledge of English to contribute to the learning environment. Overall, I believe that teaching professional courses in English is a valuable approach that can enhance the academic experience for students from diverse backgrounds."
    return example

def generate_correction_html(answer):
    # generate the correction
    generate_html(answer)
    correction_html = "<div style='background-color: #f0f0f0; padding: 30px 30px 30px 30px; word-wrap:break-word'>"
    data = extract_from_html()
    data = data.replace("&nbsp;", " ")
    correction_html += data
    correction_html += "</div><br>"
    return correction_html

def grade_answer(question, answer):
    # grade the answer
    feedback = "This is the feedback."
    return feedback



# start the app
st.title("Your English Writing Assistant🎯")
st.write("Please upload the question. We will give you some suggestions to improve your writing! 📝")

st.markdown('''<style type="text/css">
        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    </style>''', unsafe_allow_html=True)

question = st.text_area("Question*", height=100)
st.write("If there are images in the question, please upload them here.")
# upload the image
image = st.file_uploader("Upload the image (optional)", type=["png", "jpg", "jpeg"])
if image is not None:
    # convert the image to text
    image_description = image_to_text(image)
    question += " Image description: " + image_description

st.write("")
st.write("👉 **Please choose to generate the example answer or upload your own answer.** 👈")

tab1, tab2 = st.tabs(["Example generator", "Essay grader"])


# tab1: example generator
with tab1:
    st.header("Example generator 🚩")
    submitted_gen = st.button("Start generating")
    
    if submitted_gen:
        if question == "":
            st.error("Question cannot be empty.")
            st.stop()

        with st.spinner("Generating..."):
            # generate the example answer
            example = generate_example(question)
            st.write("The example answer is:")
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 30px 30px 30px 30px;'>{example}</div>", unsafe_allow_html=True)
        

# tab2: essay grader
with tab2:
    st.header("Essay grader 💯")
    st.write("Please upload your answer.")
    answer = st.text_area("Your answer", height=100)
    
    # calculate the words they used
    words = answer.split()
    num_words = len(words)
    st.write("You have used", num_words, "words.")

    st.write("")
    submitted_grader = st.button("Start grading")

    if submitted_grader:
        if num_words == 0:
            st.error("Answer cannot be empty.")
            st.stop()

        with st.spinner("Grading..."):
            # generate the correction
            correction_html = generate_correction_html(answer)
            st.write("💡 The correction:")
            st.markdown(correction_html, unsafe_allow_html=True)

            # grade the answer
            feedback = grade_answer(question, answer)
            st.write("💡 Feedback:")
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 30px 30px 30px 30px;'>{feedback}</div>", unsafe_allow_html=True)
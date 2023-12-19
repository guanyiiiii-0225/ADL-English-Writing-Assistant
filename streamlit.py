import streamlit as st
from PIL import Image

from grammar_checker.inter import generate_html
from grammar_checker.extract2 import extract_from_html
from image_to_text.demo_instructblip import generate_image_description
from exemplar_generator.inference.generate_writing import generate_writing_without_finetune

def image_to_text(images_paths):
    description_arr = []
    for image_path in images_paths:
        image = Image.open(image_path).convert('RGB')
        # convert the image to text
        image_description = "This is an image."
        description_arr.append(image_description)

    return description_arr

def extend_question(question, image_description_arr):
    # add the image descriptions to the question
    if len(image_description_arr) == 1:
        question += "\n\n"
        question += f"Figure: {image_description_arr[0]}"
    else:
        for idx, desc in enumerate(image_description_arr):
            question += "\n\n"
            question += f"Figure {idx + 1}: {desc}"
    
    return question

def generate_example(question):
    # generate the example answer
    example = generate_writing_without_finetune(question)
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
st.write("If there is an image associated with the question, kindly upload it here. For a single image, the file should be named 0001. In case of multiple images, please name them sequentially.")
# upload the image
images_paths = st.file_uploader("Upload the images (optional, accept mutilple images)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if len(images_paths) > 0:
    # convert the image to text
    image_description_arr = image_to_text(images_paths)

    # add the image description to the question
    question = extend_question(question, image_description_arr)

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
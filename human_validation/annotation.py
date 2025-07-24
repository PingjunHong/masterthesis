# streamlit_app.py
import streamlit as st
import json
import os

# set page config
DATA_PATH = 'path/to/your/data.jsonl'
SAVE_PATH = 'path/to/save/annotations.jsonl'

# load data
@st.cache_data
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return [ex for ex in data if ex.get("validation") == "No"]

# save annotation
def save_annotation(example, q1, q2):
    annotated = dict(example)
    annotated["validation_question_1"] = q1
    annotated["validation_question_2"] = q2
    with open(SAVE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(annotated, ensure_ascii=False) + '\n')

if "index" not in st.session_state:
    st.session_state.index = 0

data = load_data(DATA_PATH)
total = len(data)

st.title("GPT-4o Explanation Validator")
st.markdown("You are annotating only examples where `validation == No`.")

if st.session_state.index >= total:
    st.success("🎉 Annotation complete!")
    st.stop()

ex = data[st.session_state.index]

st.markdown(f"### Pair ID: `{ex['pairID']}`")
st.markdown(f"**Gold Label**: `{ex['gold_label']}`")
st.markdown(f"**Taxonomy**: `{ex['taxonomy']}`")
st.markdown("#### Premise")
st.write(ex["premise"])
st.markdown("#### Hypothesis")
st.write(ex["hypothesis"])
st.markdown("#### Explanation")
st.info(ex["explanation"])

# Q1: Label Fit
q1 = st.radio("1️⃣ Does the explanation fit the gold label?", ["Yes", "No", "Unsure"], key="q1")

# Q2: Taxonomy Fit
q2 = st.radio("2️⃣ Does the explanation fit the taxonomy?", ["Yes", "No", "Unsure"], key="q2")

if st.button("✅ Submit and Next"):
    save_annotation(ex, q1, q2)
    st.session_state.index += 1
    st.rerun()

st.progress((st.session_state.index + 1) / total)
st.text(f"Progress: {st.session_state.index + 1} / {total}")

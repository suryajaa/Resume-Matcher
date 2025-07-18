import streamlit as st
from resume_parser import extract_text_from_pdf
from job_matcher import load_embeddings, get_top_matches, load_job_descriptions

st.set_page_config(page_title="Smart Resume Matcher", layout="wide")
st.title("📄 Smart Resume Matcher")

# Upload PDF resume
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Resume uploaded successfully!")

    # Extract resume text
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text_from_pdf("temp_resume.pdf")

    # Load job data and embeddings
    with st.spinner("Loading job data and matching..."):
        job_df = load_job_descriptions()
        embeddings = load_embeddings()
        top_matches = get_top_matches(resume_text, job_df, embeddings)

    st.subheader("Top Matching Jobs 🔍")

    for idx, row in top_matches.iterrows():
        st.markdown(f"### 💼 {row['Job Title']}")
        st.markdown(f"- **Similarity Score**: `{row['similarity']:.2f}`")
        with st.expander("📄 Click to view job description"):
            st.markdown(row['Job Description'])
        st.markdown("---")

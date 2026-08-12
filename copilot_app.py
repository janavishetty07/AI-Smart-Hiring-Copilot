import streamlit as st
import hashlib
import json
import requests
import re
import os
from pypdf import PdfReader
from docx import Document
import io
import pandas as pd

# --- ESTABLISH ARCHITECTURAL SANITY & THEME ---
st.set_page_config(page_title="AI-Driven Smart Hiring Platform with Candidate Matching Copilot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; color: #ffffff; }
    .sidebar .sidebar-content { background: #161b22; }
    .stButton>button { width: 100%; border-radius: 6px; background-color: #238636; color: white; }
    .warning-box { padding: 15px; background-color: #7e1a1a; border-radius: 6px; border: 1px solid #f85149; margin-bottom: 15px; }
    .metric-card { background-color: #1f2937; padding: 20px; border-radius: 8px; border: 1px solid #374151; memory-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- LOCAL STORAGE PERSISTENCE ENGINE ---
DB_FILE = "registry_db.json"

def load_persistent_registry():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_persistent_registry(registry):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(registry, f, indent=4)
    except Exception as e:
        st.error(f"Persistence Sync Error: {str(e)}")

# --- SYSTEM CACHE & REGISTRY INITIALIZATION ---
if "integrity_registry" not in st.session_state:
    st.session_state.integrity_registry = load_persistent_registry()
if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = [
        {"type": "info", "msg": "Persistence layer hydration complete."},
        {"type": "success", "msg": "Local Llama 3.2 engine online."}
    ]
if "active_jd_spec" not in st.session_state:
    st.session_state.active_jd_spec = ""
if "active_skills" not in st.session_state:
    st.session_state.active_skills = {"Python": 5, "SQL": 3, "Machine Learning": 4}

OLLAMA_URL = "http://localhost:11434/api/generate"

def stream_llama(prompt):
    """High-stability streaming engine with extended wake-up timeout buffers."""
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "options": {
            "num_predict": 800,
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=90)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                yield chunk.get("response", "")
                if chunk.get("done", False):
                    break
    except requests.exceptions.Timeout:
        yield "\n[System Timeout]: Local Llama 3.2 is preparing weights. Re-run transaction."
    except Exception as e:
        yield f"\n[Streaming Offline]: Verify background Ollama instance status. Details: {str(e)}"

        # ==========================================
# ENTERPRISE AI GUARD
# ==========================================

ALLOWED_KEYWORDS = [
    "candidate",
    "resume",
    "skill",
    "skills",
    "experience",
    "education",
    "project",
    "certification",
    "job",
    "job description",
    "jd",
    "recruitment",
    "recruit",
    "hiring",
    "interview",
    "comparison",
    "ranking",
    "talent",
    "employee",
    "promotion",
    "career",
    "python",
    "sql",
    "machine learning",
    "data science",
    "ai",
    "ml",
    "streamlit",
    "ollama",
    "llama",
    "dashboard",
    "copilot",
    "module"
]


def is_recruitment_query(question):
    question = question.lower()
    return any(keyword in question for keyword in ALLOWED_KEYWORDS)


def build_secure_prompt(context, question):
    return f"""
You are the Enterprise AI Assistant for the AI-Driven Smart Hiring Platform with Candidate Matching Copilot.

STRICT RULES

You may answer ONLY about:

• Candidate resumes
• Candidate skills
• Experience
• Education
• Certifications
• Projects
• Job descriptions
• Candidate ranking
• Candidate comparison
• Interview preparation
• Hiring recommendations
• Talent management
• Recruitment workflow
• This project's modules and features

DO NOT answer:

• Sports
• IPL
• Cricket
• Movies
• Celebrities
• Politics
• Weather
• History
• Mathematics
• Programming unrelated to recruitment
• General knowledge
• Anything outside this project

If the question is unrelated, reply ONLY:

"This question is outside the scope of the AI Recruitment & Talent Management Copilot. Please ask questions related to candidates, recruitment, resumes, job descriptions, or project modules."

Context:

{context}

Question:

{question}

Answer:
"""

# ==========================================
# LLAMA JSON HELPER
# ==========================================

def llama_json(prompt):
    """Returns structured JSON output from Llama."""
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 600
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        return json.loads(result["response"])

    except Exception as e:
        st.error(f"AI JSON Extraction Error: {e}")
        return {}

# ==========================================
# RESUME EXTRACTION HELPERS
# ==========================================

def clean_list(items):
    cleaned = []
    for item in items:
        item = re.sub(r"\s+", " ", item.strip())
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned

def extract_skills(text):
    skill_database = [
        "Python", "SQL", "R", "C", "C++", "Java",
        "Machine Learning", "Deep Learning", "Data Science", "Data Analysis", "EDA",
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
        "Matplotlib", "Seaborn",
        "Power BI", "Tableau", "Excel",
        "Streamlit", "GitHub", "Jupyter Notebook",
        "MySQL", "MongoDB",
        "AWS", "Azure"
    ]
    found = []
    text_lower = text.lower()
    for skill in skill_database:
        if skill.lower() in text_lower:
            found.append(skill)
    return clean_list(found)

def extract_section(text, headings):
    text_lower = text.lower()
    for heading in headings:
        index = text_lower.find(heading.lower())
        if index != -1:
            return text[index:index+1200]
    return ""

def extract_education(text):
    patterns = [
        "education", "academic qualification", "academic background",
        "qualification", "degree", "university"
    ]
    text_lower = text.lower()
    start = -1
    for p in patterns:
        pos = text_lower.find(p)
        if pos != -1:
            start = pos
            break

    if start == -1:
        return "Education details not detected"

    section = text[start:start+800]
    lines = section.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if len(line) > 5:
            if not any(
                x in line.lower()
                for x in ["skills", "project", "experience", "certification"]
            ):
                result.append(line)

    return " | ".join(result[:5])

def extract_projects(text):
    headings = [
        "projects", "project experience", "academic projects",
        "personal projects", "key projects"
    ]
    text_lower = text.lower()
    start = -1
    for h in headings:
        pos = text_lower.find(h)
        if pos != -1:
            start = pos
            break

    if start == -1:
        return []

    section = text[start:start+1500]
    projects = []
    for line in section.split("\n"):
        line = line.strip()
        if len(line) > 8:
            if not any(
                x in line.lower()
                for x in ["skills", "education", "certification", "experience"]
            ):
                projects.append(line)

    return clean_list(projects[:10])

def extract_certifications(text):
    headings = [
        "certifications", "certificates", "certification",
        "courses", "training"
    ]
    text_lower = text.lower()
    start = -1
    for h in headings:
        pos = text_lower.find(h)
        if pos != -1:
            start = pos
            break

    if start == -1:
        return []

    section = text[start:start+1000]
    certifications = []
    for line in section.split("\n"):
        line = line.strip()
        if len(line) > 5:
            certifications.append(line)

    return clean_list(certifications[:10])

def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(?:years|year|yrs)", text.lower())
    if matches:
        return max(int(x) for x in matches)
    return 0

# --- SIDEBAR WORKSPACE CONTROL NAVIGATION ---
st.sidebar.title("🤖 AI-Driven Smart Hiring Platform with Candidate Matching Copilot")
st.sidebar.markdown("---")
module = st.sidebar.radio(
    "Navigation Workspace",
    [
        "Executive Dashboard & Insights", 
        "Job Description Intelligence", 
        "Resume Core & Integrity Shield",
        "Candidate Ranking Matrix",
        "Interview Engineering Engine",
        "Comparison AI Grid",
        "Automated Communication Hub",
        "Talent Management Strategy"
    ]
)

# ==========================================
# EXECUTIVE DASHBOARD & INSIGHTS
# ==========================================

if module == "Executive Dashboard & Insights":

    st.title("📊 Executive Recruitment Intelligence Dashboard")
    st.markdown("---")

    # LOAD LIVE REGISTRY DATA
    registry = st.session_state.integrity_registry
    total_profiles = len(registry)

    verified_candidates = [
        candidate for candidate in registry.values()
        if not candidate.get("flagged", False)
    ]

    flagged_candidates = [
        candidate for candidate in registry.values()
        if candidate.get("flagged", False)
    ]

    verified_count = len(verified_candidates)
    flagged_count = len(flagged_candidates)

    threat_percentage = (
        round((flagged_count / total_profiles) * 100, 2)
        if total_profiles > 0 else 0
    )

    # SCORE ANALYTICS
    candidate_scores = [
        candidate.get("overall_score", 0)
        for candidate in verified_candidates
    ]

    average_score = (
        round(sum(candidate_scores) / len(candidate_scores), 2)
        if candidate_scores else 0
    )

    top_candidate = None
    if verified_candidates:
        top_candidate = max(
            verified_candidates,
            key=lambda x: x.get("overall_score", 0)
        )

    # SKILL INTELLIGENCE
    skill_distribution = {}
    for candidate in verified_candidates:
        for skill in candidate.get("skills", []):
            skill_distribution[skill] = skill_distribution.get(skill, 0) + 1

    # EXPERIENCE ANALYTICS
    total_experience = sum(
        [float(candidate.get("experience_years", 0)) for candidate in verified_candidates]
    )

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Resumes Processed", total_profiles)

    with col2:
        st.metric("✅ Verified Candidates", verified_count)

    with col3:
        st.metric("⭐ Average AI Match Score", f"{average_score}%")

    with col4:
        if top_candidate:
            st.metric("🏆 Top Ranked Candidate", top_candidate.get("name", "Unknown"))
        else:
            st.metric("🏆 Top Ranked Candidate", "Not Available")

    st.markdown("---")

    # SECONDARY ANALYTICS
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📈 Recruitment Pipeline Overview")
        pipeline_data = {
            "Total Profiles": total_profiles,
            "Verified": verified_count,
            "Flagged": flagged_count
        }
        st.bar_chart(pipeline_data)

    with col6:
        st.subheader("🛡️ Integrity Shield Status")
        st.metric("Duplicate/Fraud Detection", f"{flagged_count} Profiles")
        st.metric("Threat Exposure", f"{threat_percentage}%")

    st.markdown("---")

    # SKILL DISTRIBUTION
    st.subheader("🧠 Candidate Skill Intelligence")
    if skill_distribution:
        st.bar_chart(skill_distribution)
    else:
        st.info("Skill analytics will appear after resume processing.")

    st.markdown("---")

    # EXPERIENCE INSIGHTS
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("💼 Talent Experience Pool")
        st.metric("Total Candidate Experience", f"{total_experience} Years")

    with col8:
        st.subheader("🎯 Active Job Requirement")
        if st.session_state.active_jd_spec:
            st.success("Job Description Loaded")
            st.write(st.session_state.active_jd_spec[:300] + "...")
        else:
            st.warning("No Job Description Available")

    st.markdown("---")

    # SYSTEM LOGS
    st.subheader("📋 System Audit Logs & Activity Feed")
    for entry in reversed(st.session_state.recent_logs[-8:]):
        if entry["type"] == "success":
            st.success(entry["msg"])
        elif entry["type"] == "warning":
            st.warning(entry["msg"])
        else:
            st.info(entry["msg"])

# ==========================================
# JOB DESCRIPTION INTELLIGENCE
# ==========================================
elif module == "Job Description Intelligence":
    st.title("🎯 Job Profile & Skill Gap Strategy")
    st.markdown("---")
    
    jd_mode = st.radio("Choose Input Type:", ["Generate via Job Title Template", "Paste Existing Raw Job Description"])
    
    if jd_mode == "Generate via Job Title Template":
        raw_title = st.text_input("Target Enterprise Role Title", placeholder="e.g., Lead AI Engineer")
        if st.button("Generate Enterprise Specification Matrix"):
            if raw_title:
                st.subheader("Streamed Profile Structure")
                placeholder = st.empty()
                response_accumulator = ""
                prompt = f"Create a concise, professional, structured Job Description outline for a {raw_title}. Include Core Responsibilities and Required Toolsets."
                for token in stream_llama(prompt):
                    response_accumulator += token
                    placeholder.markdown(response_accumulator)
                st.session_state.active_jd_spec = response_accumulator
                
                words = [word.strip(",.()").capitalize() for word in raw_title.split() if len(word) > 3]
                for w in words:
                    if w not in ["Lead", "Senior", "Engineer", "Role", "With", "Developer"]:
                        st.session_state.active_skills[w] = st.session_state.active_skills.get(w, 0) + 1
            else:
                st.warning("Please specify a target role first.")
                
    else:
        pasted_jd = st.text_area("Paste Raw Job Description Text here", height=250)
        if st.button("Analyze and Extract Skill Vectors"):
            if pasted_jd:
                st.subheader("Extracted Job Context Analytics")
                placeholder = st.empty()
                response_accumulator = ""
                prompt = f"Analyze this Job Description and list out the exact Top 5 core tools/skills required in a clean bulleted list:\n\n{pasted_jd}"
                for token in stream_llama(prompt):
                    response_accumulator += token
                    placeholder.markdown(response_accumulator)
                st.session_state.active_jd_spec = pasted_jd
                st.success("Skill matrices successfully routed to the Executive Dashboard pipeline.")
            else:
                st.warning("Please paste structural text before executing analysis.")

# ==========================================
# MODULE 3
# RESUME CORE & INTEGRITY SHIELD
# ==========================================

elif module == "Resume Core & Integrity Shield":

    st.title("🛡️ Resume Intelligence & Candidate Profile Engine")
    st.markdown("---")

    st.subheader("👤 Candidate Information")

    col1, col2 = st.columns(2)

    with col1:
        c_name = st.text_input("Candidate Full Name")
        c_email = st.text_input("Candidate Email")
        c_phone = st.text_input("Phone Number")

    with col2:
        c_role = st.selectbox(
            "Applying Role",
            ["Data Analyst", "Python Developer", "AI Engineer", "ML Engineer", "Other"]
        )
        c_experience = st.selectbox(
            "Experience Level",
            ["Fresher", "1-2 Years", "3-5 Years", "5+ Years"]
        )

    uploaded_resume = st.file_uploader(
        "📄 Upload Resume",
        type=["pdf", "docx", "txt"]
    )

    def extract_resume_text(file):
        text = ""
        extension = file.name.split(".")[-1].lower()
        try:
            if extension == "txt":
                text = file.read().decode("utf-8")
            elif extension == "pdf":
                pdf_reader = PdfReader(io.BytesIO(file.read()))
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            elif extension == "docx":
                document = Document(io.BytesIO(file.read()))
                for paragraph in document.paragraphs:
                    text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Resume extraction failed: {e}")
        return text

    if st.button("🚀 Process Candidate Profile"):
        if not c_name or not c_email or not uploaded_resume:
            st.warning("Please enter candidate details and upload resume.")
        else:
            raw_text = extract_resume_text(uploaded_resume)

            if not raw_text.strip():
                st.error("Resume text extraction failed.")
            else:
                with st.expander("📄 Extracted Resume Preview"):
                    st.write(raw_text[:4000])

                # Execute Extractions
                skills = extract_skills(raw_text)
                education = extract_education(raw_text)
                projects = extract_projects(raw_text)
                certifications = extract_certifications(raw_text)
                experience_years = extract_experience(raw_text)

                # Generate Identifier Hash for Integrity Check
                sanitized_body = re.sub(r'\b(?:' + re.escape(c_name) + r'|' + re.escape(c_email) + r')\b', '', raw_text, flags=re.IGNORECASE)
                normalized_body = "".join(sanitized_body.split()).lower()
                credential_hash = hashlib.sha256(normalized_body.encode('utf-8')).hexdigest()

                # Store Candidate Profile
                st.session_state.integrity_registry[credential_hash] = {
                    "name": c_name,
                    "email": c_email,
                    "phone": c_phone,
                    "role": c_role,
                    "experience_level": c_experience,
                    "resume_text": raw_text,
                    "skills": skills,
                    "education": education,
                    "projects": projects,
                    "certifications": certifications,
                    "experience_years": experience_years,
                    "skill_score": 0,
                    "experience_score": 0,
                    "education_score": 0,
                    "overall_score": 0,
                    "flagged": False
                }

                save_persistent_registry(st.session_state.integrity_registry)
                st.success("Candidate Profile Successfully Processed and Indexed!")

                # AI RESUME INTELLIGENCE SUMMARY
                st.markdown("---")
                st.subheader("🤖 AI Resume Intelligence Summary")

                summary_prompt = f"""
Provide a concise professional recruitment summary for candidate {c_name}:
1. Experience & Role Fit
2. Technical Skills Matrix
3. Academic & Project Highlights

Resume Context:
{raw_text[:2500]}
"""

                response_box = st.empty()
                summary_output = ""

                for token in stream_llama(summary_prompt):
                    summary_output += token
                    response_box.markdown(summary_output)

# ==========================================
# MODULE 4: CANDIDATE RANKING MATRIX
# ==========================================

elif module == "Candidate Ranking Matrix":

    st.title("📊 Matrix Matching & Candidate Score Ranking")
    st.markdown("---")

    candidates = {
        k: v for k, v in st.session_state.integrity_registry.items()
        if not v.get("flagged", False)
    }

    if not candidates:
        st.info("No verified candidates available. Upload resumes in Resume Core module first.")

    elif not st.session_state.active_jd_spec:
        st.warning("Please create or upload a Job Description first.")

    else:

        if "jd_requirements" not in st.session_state:

            jd_prompt = f"""
Extract job requirements.
Return ONLY JSON.

{{
    "skills": [],
    "experience_years": 0,
    "education": ""
}}

Job Description:

{st.session_state.active_jd_spec}
"""

            st.session_state.jd_requirements = llama_json(jd_prompt)


        jd = st.session_state.jd_requirements


        def skill_match(candidate_skills, required_skills):

            if not required_skills:
                return 0

            candidate_skills = [
                str(x).lower().strip()
                for x in candidate_skills
            ]

            required_skills = [
                str(x).lower().strip()
                for x in required_skills
            ]

            matches = len(
                set(candidate_skills) &
                set(required_skills)
            )

            return round(
                (matches / len(required_skills)) * 100,
                2
            )


        def experience_match(candidate_exp, required_exp):

            try:
                candidate_exp = float(candidate_exp)
                required_exp = float(required_exp)

            except:
                return 0


            if required_exp == 0:
                return 100


            score = (
                candidate_exp /
                required_exp
            ) * 100


            return min(
                round(score, 2),
                100
            )


        def education_match(candidate, required):

            candidate = str(candidate).lower()
            required = str(required).lower()


            if not required:
                return 100


            if required in candidate:
                return 100


            return 50



        ranking = []


        for key, candidate in candidates.items():

            skill_score = skill_match(
                candidate.get("skills", []),
                jd.get("skills", [])
            )


            exp_score = experience_match(
                candidate.get("experience_years", 0),
                jd.get("experience_years", 0)
            )


            edu_score = education_match(
                candidate.get("education", ""),
                jd.get("education", "")
            )


            final_score = round(
                (
                    skill_score * 0.45
                    +
                    exp_score * 0.35
                    +
                    edu_score * 0.20
                ),
                2
            )


            candidate["skill_score"] = skill_score
            candidate["experience_score"] = exp_score
            candidate["education_score"] = edu_score
            candidate["overall_score"] = final_score


            ranking.append(
                {
                    "Candidate": candidate.get(
                        "name",
                        "Unknown"
                    ),

                    "Email": candidate.get(
                        "email",
                        "Not Available"
                    ),

                    "Skill Fit %": skill_score,

                    "Experience Fit %": exp_score,

                    "Education Fit %": edu_score,

                    "Overall Score %": final_score
                }
            )



        save_persistent_registry(
            st.session_state.integrity_registry
        )


        df = pd.DataFrame(ranking)



        if df.empty:

            st.warning(
                "No candidates available for ranking."
            )


        else:

            df = df.sort_values(
                by="Overall Score %",
                ascending=False
            )


            df.reset_index(
                drop=True,
                inplace=True
            )


            df.index += 1
            df.index.name = "Rank"



            top = df.iloc[0]


            c1, c2, c3, c4 = st.columns(4)


            with c1:
                st.metric(
                    "🥇 Best Candidate",
                    top["Candidate"]
                )


            with c2:
                st.metric(
                    "Highest Score",
                    f'{top["Overall Score %"]}%'
                )


            with c3:
                st.metric(
                    "Candidates Ranked",
                    len(df)
                )


            with c4:
                st.metric(
                    "Average Score",
                    f'{round(df["Overall Score %"].mean(),2)}%'
                )



            st.markdown("---")


            st.subheader(
                "🏆 Candidate Ranking Table"
            )


            st.dataframe(
                df,
                use_container_width=True
            )



            csv = df.to_csv().encode(
                "utf-8"
            )


            st.download_button(
                "📥 Download Ranking Report",
                csv,
                "candidate_ranking.csv",
                "text/csv",
                key="ranking_download"
            )



            st.markdown("---")



            selected = st.selectbox(
                "Select Candidate for AI Analysis",
                df["Candidate"].tolist(),
                key="candidate_analysis"
            )



            if st.button(
                "Generate AI Hiring Explanation",
                key="generate_explanation"
            ):


                profile = next(
                    v for v in candidates.values()
                    if v.get("name") == selected
                )


                context = f"""

Candidate Resume:

{profile.get("resume_text","")[:2000]}


Candidate Scores:

Skills:
{profile.get("skill_score",0)}%


Experience:
{profile.get("experience_score",0)}%


Education:
{profile.get("education_score",0)}%


Job Description:

{st.session_state.active_jd_spec}

"""


                prompt = build_secure_prompt(
                    context,
                    """
Explain this candidate ranking.

Provide:

1. Candidate strengths

2. Weak areas

3. Hiring recommendation
"""
                )


                box = st.empty()

                output = ""


                for token in stream_llama(prompt):

                    output += token
                    box.markdown(output)
# ==========================================
# MODULE 5: HEAD-TO-HEAD CANDIDATE COMPARISON
# ==========================================

elif module == "Comparison AI Grid":

    st.title("⚖️ Head-to-Head Candidate Comparison AI")
    st.markdown("---")


    candidates = {
        v["name"]: v
        for k, v in st.session_state.integrity_registry.items()
        if not v.get("flagged", False)
    }


    if len(candidates) < 2:

        st.info("Minimum two verified candidates required.")


    else:

        names = list(candidates.keys())


        col1, col2, col3 = st.columns(3)


        with col1:

            cand_a = st.selectbox(
                "Candidate A",
                names,
                index=0
            )


        with col2:

            cand_b = st.selectbox(
                "Candidate B",
                names,
                index=1
            )


        with col3:

            optional = [
                "None"
            ] + [
                x for x in names
                if x not in [cand_a, cand_b]
            ]


            cand_c = st.selectbox(
                "Candidate C (Optional)",
                optional
            )



        selected = [
            cand_a,
            cand_b
        ]


        if cand_c != "None":

            selected.append(cand_c)



        comparison = []



        for feature in [
            "Skills",
            "Experience",
            "Education",
            "Projects",
            "Certifications",
            "Overall Score"
        ]:


            row = {
                "Attribute": feature
            }


            for name in selected:


                data = candidates[name]


                if feature == "Skills":

                    value = ", ".join(
                        data.get(
                            "skills",
                            []
                        )
                    )


                elif feature == "Experience":

                    value = (
                        str(
                            data.get(
                                "experience_years",
                                0
                            )
                        )
                        + " Years"
                    )


                elif feature == "Education":

                    value = data.get(
                        "education",
                        "Not Available"
                    )


                elif feature == "Projects":

                    value = ", ".join(
                        data.get(
                            "projects",
                            []
                        )
                    )


                elif feature == "Certifications":

                    value = ", ".join(
                        data.get(
                            "certifications",
                            []
                        )
                    )


                else:

                    value = str(
                        data.get(
                            "overall_score",
                            "Not Ranked"
                        )
                    )


                row[name] = value


            comparison.append(row)



        comp_df = pd.DataFrame(
            comparison
        )


        st.dataframe(
            comp_df,
            use_container_width=True
        )



        st.markdown("---")



        if st.button(
            "🧠 Generate Llama Hiring Verdict"
        ):


            context = ""


            for name in selected:

                context += f"""

Candidate:
{name}

Resume:
{candidates[name].get("resume_text","")[:1500]}

"""



            secure_context = f"""

Job Description:

{st.session_state.active_jd_spec[:2000]}


Candidate Information:

{context}

"""



            prompt = build_secure_prompt(
                secure_context,

                """
Compare these candidates for this role.

Provide:

1. Candidate Strength Comparison

2. Technical Advantage

3. Hiring Risk

4. Final Recommended Candidate
"""
            )



            output = ""

            box = st.empty()



            for token in stream_llama(prompt):

                output += token

                box.markdown(output)
# ==========================================
# MODULE 6: INTERVIEW ENGINEER
# ==========================================

elif module == "Interview Engineering Engine":

    st.title("🛠️ AI Interview Script Engineer")
    st.markdown("---")


    candidates = [
        v for k, v in st.session_state.integrity_registry.items()
        if not v.get("flagged", False)
    ]


    if not candidates:

        st.info("Upload candidate profiles first.")


    else:

        names = [
            c["name"] for c in candidates
        ]


        selected = st.selectbox(
            "Select Candidate",
            names
        )


        candidate = next(
            c for c in candidates
            if c["name"] == selected
        )


        interview_type = st.selectbox(
            "Interview Focus",
            [
                "Full Stack",
                "Data Science",
                "Machine Learning",
                "Software Engineering",
                "Data Analyst"
            ]
        )



        if st.button(
            "Generate Personalized Interview Kit"
        ):


            context = f"""

Candidate Name:

{candidate.get("name","")}


Candidate Resume:

{candidate.get("resume_text","")[:2500]}


Job Description:

{st.session_state.active_jd_spec}


Interview Focus:

{interview_type}

"""



            prompt = build_secure_prompt(
                context,

                """
Create a personalized interview evaluation kit.

Generate:

## Behavioral Questions

## Technical Questions

## Coding Challenge

## Expected Answer

## Evaluation Criteria
"""
            )



            output = ""

            box = st.empty()



            for token in stream_llama(prompt):

                output += token

                box.markdown(output)

# ==========================================
# MODULE 7: RESUME CHAT + EMAIL HUB
# ==========================================

elif module == "Automated Communication Hub":

    st.title("💬 Resume Chat & Automated Email Hub")
    st.markdown("---")

    candidates = {
        v["name"]: v for k, v in st.session_state.integrity_registry.items()
        if not v.get("flagged", False)
    }

    if candidates:

        selected = st.selectbox(
            "Select Candidate",
            list(candidates.keys())
        )

        candidate = candidates[selected]


        # ==============================
        # RESUME INTELLIGENCE CHAT
        # ==============================

        st.subheader("📄 Resume Intelligence Chat")

        st.info(
            "🔒 Enterprise AI Guard Active - "
            "This assistant answers only candidate, recruitment, "
            "resume, job description and project-related questions."
        )


        question = st.text_input("Ask about candidate")


        if question:

            if not is_recruitment_query(question):

                st.warning(
                    "❌ This question is outside the scope of the "
                    "AI-Driven Smart Hiring Platform with Candidate Matching Copilot.\n\n"
                    "Please ask questions related to candidates, "
                    "resumes, recruitment, job descriptions or project modules."
                )


            else:

                context = f"""
Candidate Name:
{candidate.get("name","")}

Role:
{candidate.get("role","")}

Resume:
{candidate.get("resume_text","")[:3000]}

Job Description:
{st.session_state.active_jd_spec}
"""


                prompt = build_secure_prompt(
                    context,
                    question
                )


                answer = ""
                box = st.empty()


                for token in stream_llama(prompt):
                    answer += token
                    box.markdown(answer)



        # ==============================
        # EMAIL GENERATION HUB
        # ==============================

        st.markdown("---")

        st.subheader("✉️ Candidate Communication")


        mail_type = st.selectbox(
            "Email Purpose",
            [
                "Interview Invitation",
                "Offer Letter",
                "Rejection",
                "Follow Up"
            ]
        )


        if st.button("Generate Email"):

            prompt = f"""
Generate a professional HR email.

Candidate:
{selected}

Email Purpose:
{mail_type}

Create a personalized and professional recruiter email.
"""


            email = ""

            box = st.empty()


            for token in stream_llama(prompt):

                email += token
                box.markdown(email)


            import urllib.parse


            mail_link = (
                "mailto:"
                + candidate["email"]
                + "?body="
                + urllib.parse.quote(email)
            )


            st.markdown(
                f"[📨 Open Email Client]({mail_link})"
            )
# ==========================================
# MODULE 8: TALENT MANAGEMENT STRATEGY
# ==========================================

elif module == "Talent Management Strategy":

    st.title("🚀 AI Talent Growth & Retention Strategy")
    st.markdown("---")


    candidates = [
        v for k, v in st.session_state.integrity_registry.items()
        if not v.get("flagged", False)
    ]


    if not candidates:

        st.info("Upload candidate profiles first.")


    else:

        selected = st.selectbox(
            "Select Candidate",
            [c["name"] for c in candidates]
        )


        candidate = next(
            c for c in candidates
            if c["name"] == selected
        )



        if st.button(
            "Generate AI Career Roadmap"
        ):


            context = f"""

Candidate Profile:

{candidate.get("resume_text","")[:2500]}


Candidate Name:

{candidate.get("name","")}


Job Role:

{candidate.get("role","")}

"""



            prompt = build_secure_prompt(
                context,

                """
Create an enterprise talent management plan.

Generate:

1. 30 Day Plan

2. 60 Day Plan

3. 90 Day Plan

4. Skill Improvement Roadmap

5. Retention Strategy

6. Promotion Readiness
"""
            )



            output = ""

            box = st.empty()



            for token in stream_llama(prompt):

                output += token

                box.markdown(output)
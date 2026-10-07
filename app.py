"""
========================================================================================
STUDENT LEARNING BEHAVIOUR ANALYSIS & ENGINEERING GUIDANCE PLATFORM
========================================================================================
A comprehensive Machine Learning and Engineering Guidance System featuring:
1. Student Behaviour Analyzer (K-Means Clustering & Pedagogical Insights)
2. Personalized Learning Roadmap (Skill-tailored learning pathways & progress tracking)
3. Engineering Domain Explorer (Cross-disciplinary engineering specialization hub)
4. Course & Resource Recommendations (Hierarchical 4-level fallback recommendation engine)
========================================================================================
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================================================
# 1. PAGE CONFIGURATION
# ======================================================================================

st.set_page_config(
    page_title="Student Learning Behaviour Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================================
# 2. CUSTOM DASHBOARD CSS (MODERN, PROFESSIONAL, FINAL-YEAR PROJECT UI)
# ======================================================================================

st.markdown(
    """
    <style>
    /* Global Typography & Font Smoothing */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar Background and Header */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Modern Navigation: Hide default circular radio indicators */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Navigation item card container */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        color: #94a3b8 !important;
    }

    /* Hover effect */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(99, 102, 241, 0.12) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        color: #f1f5f9 !important;
        transform: translateX(4px);
    }

    /* Highlight active navigation card */
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.35) 0%, rgba(99, 102, 241, 0.45) 100%) !important;
        border: 1px solid #6366f1 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
    }

    /* Metric Card Enhancements */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    /* Primary Button Styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        margin-bottom: 12px;
    }

    /* Custom Progress Bar */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5 0%, #22c55e 100%) !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================================================
# 3. DIRECTORY RESOLUTION & RESILIENT RESOURCE LOADERS
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parent

def resolve_path(rel_path: str) -> Path:
    """
    Search for a required asset path across multiple prospective locations
    to guarantee flawless cross-environment compatibility.
    """
    candidates = [
        BASE_DIR / rel_path,
        Path.cwd() / rel_path,
        Path(r"F:\elamathi\New folder\ds project") / rel_path,
        Path.home() / "Downloads" / rel_path
    ]
    for c in candidates:
        if c.exists():
            return c
    return BASE_DIR / rel_path


@st.cache_data
def load_project_data():
    """
    Safely load all project CSV files with complete error trapping and fallback structures.
    """
    # 1. Engineering domains
    domain_path = resolve_path("resources/engineering_domains.csv")
    if domain_path.exists():
        engineering_domains = pd.read_csv(domain_path)
    else:
        st.warning("⚠️ Warning: resources/engineering_domains.csv not found. Using built-in baseline catalog.")
        engineering_domains = pd.DataFrame(columns=["branch", "domain", "level", "description", "career"])

    # 2. Course resources
    course_path = resolve_path("resources/course_resources.csv")
    if course_path.exists():
        course_resources = pd.read_csv(course_path)
    else:
        st.warning("⚠️ Warning: resources/course_resources.csv not found.")
        course_resources = pd.DataFrame(columns=["domain", "course", "provider", "level", "cost", "resource_url"])

    # 3. Clustered students dataset
    students_path = resolve_path("report/clustered_students.csv")
    if students_path.exists():
        students = pd.read_csv(students_path)
    else:
        # Fallback synthetic frame to provide reasonable input bounds
        students = pd.DataFrame({
            "StudyHours": [5.0, 20.0, 44.0],
            "Attendance": [60.0, 80.0, 100.0],
            "Resources": [0.0, 1.0, 2.0],
            "Extracurricular": [0.0, 0.5, 1.0],
            "Motivation": [0.0, 1.0, 2.0],
            "Internet": [0.0, 1.0, 1.0],
            "OnlineCourses": [0.0, 10.0, 20.0],
            "Discussions": [0.0, 0.6, 1.0],
            "AssignmentCompletion": [50.0, 75.0, 100.0],
            "EduTech": [0.0, 0.7, 1.0],
            "StressLevel": [0.0, 1.3, 2.0]
        })

    # 4. Cluster profiles
    profiles_path = resolve_path("report/cluster_profiles.csv")
    if profiles_path.exists():
        cluster_profiles = pd.read_csv(profiles_path, index_col=0)
    else:
        cluster_profiles = pd.DataFrame()

    # 5. Teaching recommendations
    recs_path = resolve_path("report/teaching_recommendations.csv")
    if recs_path.exists():
        teaching_recommendations = pd.read_csv(recs_path)
    else:
        teaching_recommendations = pd.DataFrame({
            "BehaviourCluster": [0, 1],
            "Recommendations": [
                "Encourage regular attendance and collaborative learning. Use milestone-based assignment checkpoints.",
                "Encourage structured independent study routines, active doubt clearance, and digital connectivity resources."
            ]
        })

    return engineering_domains, course_resources, students, cluster_profiles, teaching_recommendations


@st.cache_resource
def load_ml_models():
    """
    Safely load trained StandardScaler and K-Means clustering artifacts.
    """
    scaler_path = resolve_path("models/student_behaviour_scaler.pkl")
    kmeans_path = resolve_path("models/student_behaviour_clustering.pkl")

    scaler = None
    kmeans = None

    if scaler_path.exists() and kmeans_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            kmeans = joblib.load(kmeans_path)
        except Exception as e:
            st.error(f"❌ Error loading ML model artifacts: {e}")
    else:
        st.error("❌ Model artifacts (scaler or k-means) could not be located in models/ directory.")

    return scaler, kmeans


# Load resources
engineering_domains, course_resources, students, cluster_profiles, teaching_recommendations = load_project_data()
scaler, kmeans = load_ml_models()


# ======================================================================================
# 4. CURATED ROADMAP CATALOG
# ======================================================================================

ROADMAP_CATALOG = {
    "Computer Science Engineering": {
        "Python Programming": [
            ("Python Basics", "Beginner", 5, "Simple Calculator", "https://docs.python.org/3/tutorial/"),
            ("Variables and Data Types", "Beginner", 4, "Student Record Manager", "https://docs.python.org/3/tutorial/introduction.html"),
            ("Conditional Statements and Loops", "Beginner", 6, "Number Guessing Game", "https://docs.python.org/3/tutorial/controlflow.html"),
            ("Functions and Collections", "Intermediate", 8, "Contact Book", "https://docs.python.org/3/tutorial/datastructures.html"),
            ("Object Oriented Programming", "Intermediate", 8, "Library Management System", "https://docs.python.org/3/tutorial/classes.html"),
            ("File Handling and Exceptions", "Intermediate", 6, "Student Data Manager", "https://docs.python.org/3/tutorial/inputoutput.html"),
            ("Pandas and NumPy", "Advanced", 12, "Student Data Analysis", "https://pandas.pydata.org/docs/"),
            ("Python Application Project", "Advanced", 15, "Complete Python Application", "https://docs.streamlit.io/")
        ],
        "Data Science and Machine Learning": [
            ("Python Fundamentals", "Beginner", 8, "Basic Data Analysis", "https://www.kaggle.com/learn/python"),
            ("NumPy", "Beginner", 6, "Numerical Analysis", "https://numpy.org/learn/"),
            ("Pandas", "Beginner", 8, "Student Dataset Analysis", "https://pandas.pydata.org/docs/"),
            ("Data Cleaning", "Intermediate", 8, "Clean a Student Dataset", "https://pandas.pydata.org/docs/"),
            ("Data Visualization", "Intermediate", 8, "Student Performance Dashboard", "https://matplotlib.org/stable/tutorials/"),
            ("Statistics", "Intermediate", 10, "Statistical Analysis", "https://www.khanacademy.org/math/statistics-probability"),
            ("Machine Learning", "Advanced", 15, "Student Performance Prediction", "https://scikit-learn.org/stable/tutorial/"),
            ("Model Deployment", "Advanced", 12, "Deploy an ML Dashboard", "https://docs.streamlit.io/")
        ],
        "Web Development": [
            ("HTML Fundamentals", "Beginner", 5, "Personal Portfolio", "https://developer.mozilla.org/en-US/docs/Learn_web_development"),
            ("CSS Fundamentals", "Beginner", 8, "Responsive Portfolio", "https://developer.mozilla.org/en-US/docs/Web/CSS"),
            ("JavaScript Basics", "Beginner", 10, "Interactive Calculator", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
            ("DOM Manipulation", "Intermediate", 8, "To Do Application", "https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model"),
            ("Responsive Design", "Intermediate", 6, "Responsive Website", "https://developer.mozilla.org/en-US/docs/Learn_web_development"),
            ("Git and GitHub", "Intermediate", 5, "Portfolio Repository", "https://docs.github.com/en/get-started"),
            ("React", "Advanced", 15, "React Application", "https://react.dev/learn"),
            ("Full Stack Project", "Advanced", 20, "Full Stack Student Portal", "https://flask.palletsprojects.com/")
        ]
    },

    "Information Technology": {
        "Cloud Computing": [
            ("Networking Fundamentals", "Beginner", 8, "Network Basics Lab", "https://www.netacad.com/"),
            ("Linux Fundamentals", "Beginner", 8, "Linux Command Practice", "https://ubuntu.com/tutorials"),
            ("Cloud Concepts", "Beginner", 6, "Cloud Architecture Diagram", "https://learn.microsoft.com/training/"),
            ("Virtual Machines", "Intermediate", 8, "Virtual Server Setup", "https://learn.microsoft.com/training/"),
            ("Cloud Storage and Databases", "Intermediate", 10, "Cloud Storage Demo", "https://learn.microsoft.com/training/"),
            ("Docker Fundamentals", "Advanced", 10, "Containerize an Application", "https://docs.docker.com/get-started/"),
            ("Cloud Deployment", "Advanced", 15, "Deploy a Web Application", "https://docs.streamlit.io/")
        ],
        "Cybersecurity": [
            ("Computer and Network Basics", "Beginner", 8, "Network Diagram", "https://www.netacad.com/"),
            ("Linux Basics", "Beginner", 8, "Linux Practice Lab", "https://ubuntu.com/tutorials"),
            ("Security Fundamentals", "Intermediate", 10, "Security Awareness Guide", "https://www.cisa.gov/"),
            ("Web Security Basics", "Intermediate", 10, "Secure Web App Checklist", "https://owasp.org/www-project-top-ten/"),
            ("Cryptography Fundamentals", "Advanced", 12, "Encryption Demonstration", "https://www.nist.gov/"),
            ("Security Assessment", "Advanced", 15, "Authorized Lab Security Report", "https://portswigger.net/web-security")
        ]
    },

    "Electronics and Communication Engineering": {
        "Embedded Systems and IoT": [
            ("Basic Electrical and Electronic Concepts", "Beginner", 8, "Basic Circuit Study", "https://www.allaboutcircuits.com/textbook/"),
            ("C Programming", "Beginner", 10, "Embedded C Exercises", "https://www.learn-c.org/"),
            ("Digital Electronics", "Intermediate", 12, "Logic Gate Demonstration", "https://nptel.ac.in/"),
            ("Microcontroller Fundamentals", "Intermediate", 12, "LED Control", "https://docs.arduino.cc/"),
            ("Sensors and Actuators", "Intermediate", 10, "Temperature Monitoring", "https://docs.arduino.cc/"),
            ("IoT Communication", "Advanced", 12, "IoT Sensor Dashboard", "https://docs.arduino.cc/"),
            ("Integrated IoT Project", "Advanced", 20, "Smart Monitoring System", "https://docs.arduino.cc/")
        ],
        "VLSI Design": [
            ("Digital Logic", "Beginner", 10, "Logic Gate Circuits", "https://nptel.ac.in/"),
            ("Boolean Algebra", "Beginner", 8, "Logic Simplification", "https://nptel.ac.in/"),
            ("Combinational Circuits", "Intermediate", 12, "Adder Design", "https://nptel.ac.in/"),
            ("Sequential Circuits", "Intermediate", 12, "Counter Design", "https://nptel.ac.in/"),
            ("Verilog Fundamentals", "Advanced", 15, "Digital Module Simulation", "https://www.edaplayground.com/"),
            ("VLSI Design Project", "Advanced", 20, "RTL Design Project", "https://nptel.ac.in/")
        ]
    },

    "Electrical Engineering": {
        "Power Systems and Renewable Energy": [
            ("Electrical Fundamentals", "Beginner", 8, "Basic Circuit Calculations", "https://nptel.ac.in/"),
            ("Electrical Machines", "Beginner", 12, "Machine Study Report", "https://nptel.ac.in/"),
            ("Power Generation", "Intermediate", 10, "Power Generation Study", "https://nptel.ac.in/"),
            ("Transmission and Distribution", "Intermediate", 12, "Power Network Diagram", "https://nptel.ac.in/"),
            ("Solar Energy Fundamentals", "Intermediate", 10, "Solar System Calculation", "https://www.nrel.gov/"),
            ("Power System Protection", "Advanced", 15, "Protection System Study", "https://nptel.ac.in/"),
            ("Renewable Energy Project", "Advanced", 20, "Solar Power Design Study", "https://www.nrel.gov/")
        ],
        "PLC and Industrial Automation": [
            ("Electrical Control Fundamentals", "Beginner", 8, "Control Circuit Diagram", "https://nptel.ac.in/"),
            ("Relay Logic", "Beginner", 8, "Relay Control Simulation", "https://nptel.ac.in/"),
            ("PLC Fundamentals", "Intermediate", 12, "Motor Control Logic", "https://www.plcopen.org/"),
            ("Ladder Logic", "Intermediate", 12, "Conveyor Simulation", "https://www.plcopen.org/"),
            ("HMI and SCADA", "Advanced", 15, "Monitoring Interface", "https://www.inductiveautomation.com/"),
            ("Automation Project", "Advanced", 20, "Industrial Automation Simulation", "https://www.plcopen.org/")
        ]
    },

    "Mechanical Engineering": {
        "CAD and Product Design": [
            ("Engineering Drawing", "Beginner", 10, "2D Component Drawing", "https://nptel.ac.in/"),
            ("Engineering Design Basics", "Beginner", 8, "Mechanical Component Sketch", "https://nptel.ac.in/"),
            ("3D Modeling Fundamentals", "Intermediate", 12, "3D Mechanical Part", "https://www.solidworks.com/"),
            ("Assembly Design", "Intermediate", 12, "Mechanical Assembly", "https://www.solidworks.com/"),
            ("Engineering Analysis", "Advanced", 15, "Component Analysis Report", "https://nptel.ac.in/"),
            ("Product Design Project", "Advanced", 20, "Designed Mechanical Product", "https://www.autodesk.com/learn")
        ],
        "Robotics and Automation": [
            ("Engineering Mathematics", "Beginner", 10, "Motion Calculation Exercises", "https://nptel.ac.in/"),
            ("Basic Electronics", "Beginner", 8, "Motor Driver Study", "https://www.allaboutcircuits.com/textbook/"),
            ("Control Systems", "Intermediate", 12, "Control System Simulation", "https://nptel.ac.in/"),
            ("Sensors and Actuators", "Intermediate", 10, "Sensor Demonstration", "https://docs.arduino.cc/"),
            ("Robot Kinematics", "Advanced", 15, "Robot Motion Simulation", "https://www.ros.org/"),
            ("Robotics Project", "Advanced", 20, "Obstacle Avoiding Robot", "https://www.ros.org/")
        ]
    },

    "Civil Engineering": {
        "Structural Engineering and BIM": [
            ("Engineering Drawing", "Beginner", 8, "Building Plan Drawing", "https://nptel.ac.in/"),
            ("Building Materials", "Beginner", 8, "Material Comparison Report", "https://nptel.ac.in/"),
            ("Structural Mechanics", "Intermediate", 12, "Beam Analysis", "https://nptel.ac.in/"),
            ("AutoCAD Fundamentals", "Intermediate", 12, "Residential Floor Plan", "https://www.autodesk.com/learn"),
            ("Structural Analysis Software", "Advanced", 15, "Structural Model Study", "https://nptel.ac.in/"),
            ("BIM Fundamentals", "Advanced", 15, "Building Information Model", "https://www.autodesk.com/learn"),
            ("Civil Design Project", "Advanced", 20, "Building Design Portfolio", "https://www.autodesk.com/learn")
        ],
        "Construction Management": [
            ("Construction Materials", "Beginner", 8, "Material Selection Report", "https://nptel.ac.in/"),
            ("Construction Methods", "Beginner", 10, "Construction Process Plan", "https://nptel.ac.in/"),
            ("Quantity Estimation", "Intermediate", 12, "Building Quantity Estimate", "https://nptel.ac.in/"),
            ("Project Scheduling", "Intermediate", 12, "Construction Schedule", "https://nptel.ac.in/"),
            ("Cost Management", "Advanced", 15, "Project Cost Report", "https://nptel.ac.in/"),
            ("Construction Project", "Advanced", 20, "Project Management Plan", "https://nptel.ac.in/")
        ]
    }
}


# ======================================================================================
# 5. CORE HELPER FUNCTIONS & DATA NORMALIZATION
# ======================================================================================

def normalize_domain(domain_name: str) -> str:
    """
    Strips whitespace, lowercases, and condenses inner spaces for resilient domain matching.
    """
    if domain_name is None or pd.isna(domain_name):
        return ""
    return " ".join(str(domain_name).strip().lower().split())


def normalize_level(level_name: str) -> str:
    """
    Normalizes skill level strings into standardized 'Beginner', 'Intermediate', or 'Advanced'.
    """
    if level_name is None or pd.isna(level_name):
        return "Beginner"
    lvl_str = str(level_name).strip().lower()
    if "inter" in lvl_str:
        return "Intermediate"
    elif "adv" in lvl_str:
        return "Advanced"
    return "Beginner"


def get_all_branches() -> list[str]:
    """
    Returns unified, sorted list of all engineering branches from both CSV and ROADMAP_CATALOG.
    """
    csv_branches = set()
    if not engineering_domains.empty and "branch" in engineering_domains.columns:
        csv_branches = set(engineering_domains["branch"].dropna().str.strip().unique())
    catalog_branches = set(ROADMAP_CATALOG.keys())
    return sorted(list(csv_branches.union(catalog_branches)))


def get_branch_domains(selected_branch: str) -> list[str]:
    """
    Returns unified, sorted list of all domains for a given branch from CSV and ROADMAP_CATALOG.
    """
    branch_domains = set()
    
    # 1. From CSV
    if not engineering_domains.empty and "branch" in engineering_domains.columns and "domain" in engineering_domains.columns:
        csv_matches = engineering_domains[
            engineering_domains["branch"].astype(str).str.strip().str.lower() == selected_branch.strip().lower()
        ]["domain"].dropna().str.strip().unique()
        branch_domains.update(csv_matches)
        
    # 2. From ROADMAP_CATALOG
    if selected_branch in ROADMAP_CATALOG:
        branch_domains.update(ROADMAP_CATALOG[selected_branch].keys())
        
    return sorted(list(branch_domains)) if branch_domains else ["General Engineering"]


def generate_fallback_roadmap(domain: str) -> list[tuple]:
    """
    Requirement 3:
    Automatically generates a structured, realistic fallback roadmap for any engineering domain:
    1. <Domain> Fundamentals
    2. <Domain> Core Concepts
    3. <Domain> Tools and Technologies
    4. Applied <Domain>
    5. Advanced <Domain>
    6. <Domain> Capstone Project
    """
    clean_domain = str(domain).strip().title()
    low = clean_domain.lower()

    # Intelligent safe resource links based on technical domain context
    if any(k in low for k in ["python", "java", "code", "software", "web", "algorithm"]):
        res = "https://www.freecodecamp.org/learn/"
    elif any(k in low for k in ["cloud", "azure", "aws", "network", "cyber"]):
        res = "https://learn.microsoft.com/training/"
    elif any(k in low for k in ["data", "machine learning", "ai", "deep learning"]):
        res = "https://developers.google.com/machine-learning"
    elif any(k in low for k in ["embedded", "iot", "arduino", "microcontroller"]):
        res = "https://docs.arduino.cc/"
    elif any(k in low for k in ["cad", "revit", "civil", "solidworks", "drawing", "bim"]):
        res = "https://www.autodesk.com/learn"
    elif any(k in low for k in ["bio", "genom", "protein"]):
        res = "https://www.ncbi.nlm.nih.gov/"
    else:
        res = "https://nptel.ac.in/"

    return [
        (f"{clean_domain} Fundamentals", "Beginner", 8, f"{clean_domain} Environment Setup & Core Syntax / Principles", res),
        (f"{clean_domain} Core Concepts", "Beginner", 10, f"Foundational {clean_domain} Laboratory Exercises & Proof-of-Concept", res),
        (f"{clean_domain} Tools and Technologies", "Intermediate", 14, f"Industry Standard Toolchain & Pipeline Configuration for {clean_domain}", res),
        (f"Applied {clean_domain}", "Intermediate", 16, f"Real-World Case Study Implementation & Module Simulation", res),
        (f"Advanced {clean_domain}", "Advanced", 18, f"High-Performance System Architecture & Optimization in {clean_domain}", res),
        (f"{clean_domain} Capstone Project", "Advanced", 24, f"Comprehensive End-to-End {clean_domain} Portfolio Solution", res)
    ]


def get_domain_roadmap(branch: str, domain: str) -> tuple[list[tuple], bool]:
    """
    Retrieves the roadmap for a domain.
    Returns (modules_list, is_auto_generated).
    Never raises KeyError!
    """
    norm_d = normalize_domain(domain)

    # 1. Search in exact branch in ROADMAP_CATALOG
    if branch in ROADMAP_CATALOG:
        for cat_domain, modules in ROADMAP_CATALOG[branch].items():
            if normalize_domain(cat_domain) == norm_d:
                return (modules, False)

    # 2. Search across all branches in ROADMAP_CATALOG
    for cat_branch, branch_domains in ROADMAP_CATALOG.items():
        for cat_domain, modules in branch_domains.items():
            if normalize_domain(cat_domain) == norm_d:
                return (modules, False)

    # 3. Handle synonyms and common sub-specialization aliases
    SYNONYM_MAP = {
        "data science": ("Computer Science Engineering", "Data Science and Machine Learning"),
        "artificial intelligence and machine learning": ("Computer Science Engineering", "Data Science and Machine Learning"),
        "embedded systems": ("Electronics and Communication Engineering", "Embedded Systems and IoT"),
        "internet of things": ("Electronics and Communication Engineering", "Embedded Systems and IoT"),
        "power systems": ("Electrical Engineering", "Power Systems and Renewable Energy"),
        "renewable energy": ("Electrical Engineering", "Power Systems and Renewable Energy"),
        "plc and scada": ("Electrical Engineering", "PLC and Industrial Automation"),
        "autocad": ("Mechanical Engineering", "CAD and Product Design"),
        "solidworks": ("Mechanical Engineering", "CAD and Product Design"),
        "autocad civil": ("Civil Engineering", "Structural Engineering and BIM"),
        "revit and bim": ("Civil Engineering", "Structural Engineering and BIM"),
        "structural analysis": ("Civil Engineering", "Structural Engineering and BIM"),
    }
    if norm_d in SYNONYM_MAP:
        mapped_branch, mapped_domain = SYNONYM_MAP[norm_d]
        if mapped_branch in ROADMAP_CATALOG and mapped_domain in ROADMAP_CATALOG[mapped_branch]:
            return (ROADMAP_CATALOG[mapped_branch][mapped_domain], False)

    # 4. Return Auto-Generated Structured Roadmap (is_auto_generated = True)
    return (generate_fallback_roadmap(domain), True)


def get_domain_courses(domain: str, level: str, branch: str = None) -> tuple[list[dict], int, str]:
    """
    Requirement 4:
    Hierarchical 4-Level Recommendation Strategy:
    LEVEL 1: Exact domain + exact student level
    LEVEL 2: Show all courses/resources for that domain regardless of level
             Message: 'No exact course was mapped for this level. Showing available resources for this domain.'
    LEVEL 3: Use the roadmap resources for that domain
             Message: 'Curated course listing not found in catalog. Sourced directly from domain engineering roadmap.'
    LEVEL 4: Standard domain-specific learning resources
             Message: 'Standard accredited learning resources compiled for this engineering specialization.'
    Returns (courses_list, fallback_level_int, explanation_message)
    """
    norm_dom = normalize_domain(domain)
    norm_lvl = normalize_level(level).lower()

    if not course_resources.empty and "domain" in course_resources.columns:
        cr_clean = course_resources.copy()
        cr_clean["norm_domain"] = cr_clean["domain"].apply(normalize_domain)
        cr_clean["norm_level"] = cr_clean["level"].apply(lambda x: normalize_level(x).lower())

        # LEVEL 1: Exact domain + exact student level
        l1 = cr_clean[(cr_clean["norm_domain"] == norm_dom) & (cr_clean["norm_level"] == norm_lvl)]
        if not l1.empty:
            records = l1[["course", "provider", "level", "cost", "resource_url"]].to_dict("records")
            return (records, 1, f"Showing verified courses for {domain} at {level} level.")

        # Check domain synonyms in course resources
        SYNONYMS = {
            "data science and machine learning": ["data science", "artificial intelligence and machine learning"],
            "embedded systems and iot": ["embedded systems", "internet of things"],
            "power systems and renewable energy": ["power systems", "renewable energy"],
            "cad and product design": ["autocad", "solidworks"],
            "structural engineering and bim": ["autocad civil", "revit and bim", "structural analysis"],
            "plc and industrial automation": ["plc and scada"],
        }
        syn_list = SYNONYMS.get(norm_dom, [])
        if syn_list:
            l1_syn = cr_clean[(cr_clean["norm_domain"].isin(syn_list)) & (cr_clean["norm_level"] == norm_lvl)]
            if not l1_syn.empty:
                records = l1_syn[["course", "provider", "level", "cost", "resource_url"]].to_dict("records")
                return (records, 1, f"Showing verified courses for {domain} at {level} level.")

        # LEVEL 2: Show all courses available for that domain regardless of level
        l2 = cr_clean[cr_clean["norm_domain"] == norm_dom]
        if l2.empty and syn_list:
            l2 = cr_clean[cr_clean["norm_domain"].isin(syn_list)]

        if not l2.empty:
            records = l2[["course", "provider", "level", "cost", "resource_url"]].to_dict("records")
            return (records, 2, "No exact course was mapped for this level. Showing available resources for this domain.")

    # LEVEL 3: Sourced directly from Domain Engineering Roadmap
    roadmap_modules, is_fallback = get_domain_roadmap(branch if branch else "", domain)
    if roadmap_modules:
        level_filtered = [m for m in roadmap_modules if normalize_level(m[1]).lower() == norm_lvl]
        target_mods = level_filtered if level_filtered else roadmap_modules

        records = []
        for mod in target_mods:
            topic, mod_level, est_hrs, project, resource_url = mod
            records.append({
                "course": f"{topic} (Practical Module)",
                "provider": "Curated Engineering Roadmap" if not is_fallback else "Auto-Generated Roadmap Curriculum",
                "level": mod_level,
                "cost": "Free / Open Access",
                "resource_url": resource_url
            })
        if records:
            return (records, 3, "Curated course listing not found in catalog. Sourced directly from domain engineering roadmap.")

    # LEVEL 4: Safe / general engineering learning resources
    clean_title = str(domain).strip().title()
    l4_records = [
        {
            "course": f"{clean_title} Foundation Lectures & Courseware",
            "provider": "NPTEL / SWAYAM National Repository",
            "level": level,
            "cost": "Free",
            "resource_url": "https://nptel.ac.in/"
        },
        {
            "course": f"{clean_title} Open Courseware & Laboratory Modules",
            "provider": "MIT OpenCourseWare",
            "level": level,
            "cost": "Free",
            "resource_url": "https://ocw.mit.edu/"
        },
        {
            "course": f"{clean_title} Hands-on Engineering Practice Tracks",
            "provider": "Coursera / freeCodeCamp Engineering Hub",
            "level": level,
            "cost": "Free Audit",
            "resource_url": "https://www.freecodecamp.org/"
        }
    ]
    return (l4_records, 4, "Standard accredited learning resources compiled for this engineering specialization.")


# ======================================================================================
# 6. SIDEBAR NAVIGATION
# ======================================================================================

st.sidebar.markdown(
    """
    <div style="padding: 10px 4px 18px 4px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">
                🎓
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">
                    EduLearn AI
                </h3>
                <p style="margin: 1px 0 0 0; font-size: 0.78rem; color: #818cf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Learning Intelligence Platform
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; margin: 8px 4px 6px 4px;'>DASHBOARD MENU</p>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "🤖 Student Behaviour Analyzer",
        "🎯 Personalized Learning Roadmap",
        "🧭 Engineering Domain Explorer",
        "📚 Course & Resource Recommendations"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="padding: 6px 4px; font-size: 0.76rem; color: #64748b; line-height: 1.5;">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
            <span style="color: #22c55e; font-size: 0.9rem;">●</span>
            <span style="color: #cbd5e1; font-weight: 600;">System Active</span>
        </div>
        <p style="margin: 0;"><strong>ML Model:</strong> K-Means Clustering (k=2)</p>
        <p style="margin: 0;"><strong>Pipeline:</strong> StandardScaler + Feature Matrix</p>
        <p style="margin: 0;"><strong>Catalog:</strong> 28+ Engineering Domains</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ======================================================================================
# 7. PAGE 1: 🤖 STUDENT BEHAVIOUR ANALYZER
# ======================================================================================

if page == "🤖 Student Behaviour Analyzer":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                Unsupervised Machine Learning Engine
            </div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
                🤖 Student Behaviour Analyzer
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.98rem;">
                Cluster student learning behavior patterns using trained K-Means to identify digital engagement, study routines, and pedagogical teaching recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "💡 **Academic Clarification:** This machine learning model identifies behavioral archetypes based on study habits, "
        "engagement, and digital resource usage. It is an unsupervised pattern cluster, **not** an exam-score prediction."
    )

    st.subheader("📝 Input Student Learning Metrics")

    # 11 Learning Behaviour Features
    cluster_features = [
        "StudyHours", "Attendance", "Resources", "Extracurricular",
        "Motivation", "Internet", "OnlineCourses", "Discussions",
        "AssignmentCompletion", "EduTech", "StressLevel"
    ]

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Weekly Study Hours",
            min_value=float(students["StudyHours"].min()),
            max_value=float(students["StudyHours"].max()),
            value=float(round(students["StudyHours"].mean(), 1)),
            step=0.5,
            help="Total self-study and revision hours per week"
        )

        attendance = st.number_input(
            "Class Attendance (%)",
            min_value=float(students["Attendance"].min()),
            max_value=float(students["Attendance"].max()),
            value=float(round(students["Attendance"].mean(), 1)),
            step=0.5,
            help="Percentage of lectures attended"
        )

        resources = st.number_input(
            "Learning Resources Utilization (0 - 2)",
            min_value=float(students["Resources"].min()),
            max_value=float(students["Resources"].max()),
            value=float(round(students["Resources"].mean())),
            step=1.0,
            help="0: Minimal, 1: Moderate textbook/slides, 2: Extensive multi-source references"
        )

        extracurricular = st.selectbox(
            "Extracurricular Activity Participation",
            options=[0, 1],
            format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)",
            index=int(round(students["Extracurricular"].mean())),
            help="Participation in technical clubs, sports, or campus societies"
        )

        motivation = st.number_input(
            "Academic Motivation Level (0 - 2)",
            min_value=float(students["Motivation"].min()),
            max_value=float(students["Motivation"].max()),
            value=float(round(students["Motivation"].mean())),
            step=1.0,
            help="0: Low self-drive, 1: Moderate, 2: High self-motivation"
        )

        internet = st.selectbox(
            "High-Speed Internet Access",
            options=[1, 0],
            format_func=lambda x: "Yes (Available - 1)" if x == 1 else "No (Restricted / Offline - 0)",
            index=0,
            help="Consistent broadband or mobile high-speed connectivity"
        )

    with col2:
        online_courses = st.number_input(
            "Online Courses & Modules Completed",
            min_value=float(students["OnlineCourses"].min()),
            max_value=float(students["OnlineCourses"].max()),
            value=float(round(students["OnlineCourses"].mean(), 1)),
            step=1.0,
            help="Count of completed MOOC modules or skill tracks"
        )

        discussions = st.selectbox(
            "Discussion Forum Participation",
            options=[1, 0],
            format_func=lambda x: "Active Contributor (1)" if x == 1 else "Passive / Inactive (0)",
            index=int(round(students["Discussions"].mean())),
            help="Participation in academic forums, peer study groups, or Q&A channels"
        )

        assignment = st.number_input(
            "Assignment Completion Rate (%)",
            min_value=float(students["AssignmentCompletion"].min()),
            max_value=float(students["AssignmentCompletion"].max()),
            value=float(round(students["AssignmentCompletion"].mean(), 1)),
            step=0.5,
            help="Percentage of coursework and lab assignments submitted on time"
        )

        edutech = st.selectbox(
            "Educational Technology Adoption",
            options=[1, 0],
            format_func=lambda x: "High Adoption (LMS & Dev Tools - 1)" if x == 1 else "Basic / Traditional (0)",
            index=int(round(students["EduTech"].mean())),
            help="Active engagement with interactive coding environments and LMS platforms"
        )

        stress = st.number_input(
            "Perceived Academic Stress Level (0 - 2)",
            min_value=float(students["StressLevel"].min()),
            max_value=float(students["StressLevel"].max()),
            value=float(round(students["StressLevel"].mean())),
            step=1.0,
            help="0: Low stress, 1: Moderate stress, 2: High stress"
        )

    # Input DataFrame with explicit feature names
    input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "Resources": resources,
        "Extracurricular": float(extracurricular),
        "Motivation": motivation,
        "Internet": float(internet),
        "OnlineCourses": online_courses,
        "Discussions": float(discussions),
        "AssignmentCompletion": assignment,
        "EduTech": float(edutech),
        "StressLevel": stress
    }])

    st.divider()

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Student Behaviour", type="primary", use_container_width=True)

    if analyze_clicked:
        if scaler is None or kmeans is None:
            st.error("❌ Machine Learning pipeline is unavailable. Please verify model files in models/ folder.")
        else:
            with st.spinner("Executing StandardScaler transformation and K-Means cluster assignment..."):
                input_scaled = scaler.transform(input_df)
                predicted_cluster = int(kmeans.predict(input_scaled)[0])

            # Cluster Archetype Definitions
            cluster_meta = {
                0: {
                    "title": "Digitally Connected & Collaborative Learner",
                    "badge": "High Digital Engagement",
                    "desc": "Characterized by stable internet connectivity, active forum discussions, consistent online coursework completion, and interactive edtech utilization."
                },
                1: {
                    "title": "Traditional & Resource-Optimized Learner",
                    "badge": "Offline / Self-Paced Focus",
                    "desc": "Characterized by localized self-study patterns, traditional reference resources, high personal discipline, and focused assignment completion without reliance on live digital media."
                }
            }

            meta = cluster_meta.get(predicted_cluster, {
                "title": f"Learner Pattern Archetype {predicted_cluster}",
                "badge": "Custom Cluster",
                "desc": "Identified behavior distribution across academic effort and resource utilization."
            })

            # Polished Result Card (Requirement 7)
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
                            border: 1px solid rgba(99, 102, 241, 0.4);
                            border-radius: 14px;
                            padding: 24px;
                            margin: 20px 0 24px 0;
                            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                        <div>
                            <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #818cf8; font-weight: 700;">
                                Student Behaviour Cluster
                            </span>
                            <h2 style="margin: 4px 0 6px 0; color: #ffffff; font-size: 2.1rem; font-weight: 800;">
                                Cluster {predicted_cluster}
                            </h2>
                            <p style="margin: 0; color: #38bdf8; font-size: 1.05rem; font-weight: 600;">
                                {meta['title']}
                            </p>
                            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.92rem; max-width: 680px;">
                                {meta['desc']}
                            </p>
                        </div>
                        <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 14px 22px; text-align: center;">
                            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">Cluster Tag</span>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc; margin-top: 2px;">{meta['badge']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.subheader("📊 Cluster Behaviour Profile")
                if not cluster_profiles.empty and predicted_cluster in cluster_profiles.index:
                    profile = cluster_profiles.loc[predicted_cluster]
                    profile_df = pd.DataFrame({
                        "Behaviour Feature": profile.index,
                        "Cluster Average": profile.values.round(2),
                        "Student Input": [
                            study_hours, attendance, resources, extracurricular,
                            motivation, internet, online_courses, discussions,
                            assignment, edutech, stress
                        ]
                    })
                    st.dataframe(profile_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Cluster average profile matrix is displayed when report/cluster_profiles.csv is present.")

            with res_col2:
                st.subheader("💡 Teaching Insight & Guidance")
                rec_match = teaching_recommendations[
                    teaching_recommendations["BehaviourCluster"] == predicted_cluster
                ] if "BehaviourCluster" in teaching_recommendations.columns else pd.DataFrame()

                if not rec_match.empty:
                    rec_text = rec_match.iloc[0]["Recommendations"]
                else:
                    rec_text = (
                        "Encourage balanced learning routines, consistent attendance checkpoints, "
                        "and progressive milestones to maintain motivation and study momentum."
                    )

                st.markdown(
                    f"""
                    <div style="background: rgba(14, 165, 233, 0.08); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 18px; margin-top: 6px;">
                        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.05rem;">
                            🎯 Strategic Academic Recommendation
                        </h4>
                        <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;">
                            {rec_text}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div style="margin-top: 14px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 8px; padding: 14px;">
                        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Next Pedagogical Steps:</span>
                        <ul style="margin: 6px 0 0 16px; padding: 0; color: #cbd5e1; font-size: 0.9rem;">
                            <li>Explore personalized skill development in the <strong>Personalized Learning Roadmap</strong> tab.</li>
                            <li>Align student strengths with career opportunities via the <strong>Engineering Domain Explorer</strong>.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ======================================================================================
# 8. PAGE 2: 🎯 PERSONALIZED LEARNING ROADMAP
# ======================================================================================

elif page == "🎯 Personalized Learning Roadmap":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                Competency Engineering Pathway
            </div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
                🎯 Personalized Engineering Learning Roadmap
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.98rem;">
                Structured, practical engineering roadmap aligned with your degree branch, technical specialization, and technical mastery level.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_branches = get_all_branches()

    # Three Clean Inputs: Branch, Specialization, Skill Level (NO Study Hours per Week!)
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_branch = st.selectbox(
            "Engineering Branch",
            options=all_branches,
            index=0,
            help="Select your engineering discipline"
        )

    branch_domains = get_branch_domains(selected_branch)

    with col2:
        selected_domain = st.selectbox(
            "Technical Specialization",
            options=branch_domains,
            index=0,
            help="Select the domain track you wish to specialize in"
        )

    with col3:
        current_level = st.selectbox(
            "Current Technical Skill Level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=0,
            help="Your self-assessed competency level in this domain"
        )

    st.divider()

    # Retrieve Roadmap (Curated or Auto-Generated Fallback)
    roadmap_data, is_auto_generated = get_domain_roadmap(selected_branch, selected_domain)

    roadmap_df = pd.DataFrame(
        roadmap_data,
        columns=["topic", "level", "estimated_hours", "project", "resource"]
    )

    # Filter topics according to selected level hierarchy
    level_order = ["Beginner", "Intermediate", "Advanced"]
    selected_index = level_order.index(current_level)
    allowed_levels = level_order[:selected_index + 1]

    filtered_roadmap = roadmap_df[roadmap_df["level"].isin(allowed_levels)].reset_index(drop=True)
    if filtered_roadmap.empty:
        # If no topics match exactly (e.g., domain starts directly at Advanced), show all modules
        filtered_roadmap = roadmap_df.reset_index(drop=True)

    # Roadmap Metrics Row (Requirement 2: Replaced Estimated Weeks with Skill Level)
    total_topics = len(filtered_roadmap)
    total_hours = int(filtered_roadmap["estimated_hours"].sum())

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Learning Topics", total_topics)
    m_col2.metric("Total Estimated Hours", f"{total_hours} hrs")
    m_col3.metric("Current Level", current_level)
    m_col4.metric(
        "Curriculum Type",
        "Curated Roadmap" if not is_auto_generated else "Auto-generated"
    )

    if is_auto_generated:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; border-radius: 6px; padding: 12px 16px; margin: 16px 0;">
                <span style="color: #fbbf24; font-weight: 700;">ℹ️ Auto-generated roadmap:</span>
                <span style="color: #f1f5f9; font-size: 0.92rem;">
                    This comprehensive learning roadmap was structured automatically using core engineering competency standards.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Learning Modules Section
    st.subheader(f"📖 Learning Modules: {selected_domain}")
    st.write("Work through each module in sequence, completing the associated hands-on project.")

    completed_topics = []

    for index, row in filtered_roadmap.iterrows():
        with st.expander(f"Module {index + 1}: {row['topic']} • [{row['level']}]"):
            mod_col1, mod_col2 = st.columns([3, 1])

            with mod_col1:
                st.markdown(f"**⏱️ Estimated Duration:** `{row['estimated_hours']} hours`")
                st.markdown(f"**🛠️ Practical Project:** `{row['project']}`")
                st.link_button("🌐 Open Learning Resource", row["resource"])

            with mod_col2:
                completed = st.checkbox(
                    f"Mark Completed",
                    key=f"chk_roadmap_{normalize_domain(selected_branch)}_{normalize_domain(selected_domain)}_{index}",
                    help=f"Tick this once you have finished {row['topic']}"
                )
                if completed:
                    completed_topics.append(row["topic"])

    # Progress Tracker & Completion Percentage (Requirement 2)
    st.divider()
    st.subheader("📈 Learning Progress Tracker")

    completed_count = len(completed_topics)
    progress_val = completed_count / total_topics if total_topics > 0 else 0.0
    completion_percentage = progress_val * 100.0

    st.progress(progress_val)

    p_col1, p_col2 = st.columns(2)
    p_col1.write(f"**Status:** Completed **{completed_count}** of **{total_topics}** modules")
    p_col2.write(f"**Completion Percentage:** **{completion_percentage:.1f}%**")

    # Learning Guidance (Requirement 2: Strictly no weekly hours mentions!)
    st.divider()
    st.subheader("💡 Learning Guidance")

    if current_level == "Beginner":
        st.info(
            "🌱 **Beginner Strategy:** Master the foundational principles first. Prioritize syntax, circuit diagrams, "
            "or mathematical models before moving to complex toolchains. Ensure each foundational lab exercises runs end-to-end."
        )
    elif current_level == "Intermediate":
        st.info(
            "⚡ **Intermediate Strategy:** Shift focus to practical application, integration of multiple tools, "
            "and writing maintainable, clean code or engineering designs. Build independent portfolio artifacts."
        )
    else:
        st.info(
            "🚀 **Advanced Strategy:** Focus on systems architecture, performance tuning, failure mode analysis, "
            "and end-to-end capstone development suitable for industry demonstration or research publication."
        )


# ======================================================================================
# 9. PAGE 3: 🧭 ENGINEERING DOMAIN EXPLORER
# ======================================================================================

elif page == "🧭 Engineering Domain Explorer":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                Specialization Directory
            </div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
                🧭 Engineering Domain Explorer
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.98rem;">
                Explore career opportunities, domain descriptions, and recommended technical roadmaps for every engineering discipline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_branches = get_all_branches()

    col1, col2 = st.columns(2)

    with col1:
        selected_branch = st.selectbox(
            "Select Engineering Branch",
            options=all_branches,
            index=0
        )

    branch_domains = get_branch_domains(selected_branch)

    with col2:
        selected_domain = st.selectbox(
            "Select Technical Domain",
            options=branch_domains,
            index=0
        )

    # Fetch Domain Metadata from CSV (or intelligent fallback)
    domain_match = pd.DataFrame()
    if not engineering_domains.empty and "domain" in engineering_domains.columns:
        domain_match = engineering_domains[
            engineering_domains["domain"].apply(normalize_domain) == normalize_domain(selected_domain)
        ]

    if not domain_match.empty:
        domain_row = domain_match.iloc[0]
        desc = domain_row.get("description", f"Specialized engineering field focusing on {selected_domain} principles and applications.")
        level_val = domain_row.get("level", "Intermediate")
        career_val = domain_row.get("career", f"{selected_domain} Engineer")
    else:
        desc = f"Applied engineering specialization in {selected_domain}, emphasizing practical design, analysis, and implementation."
        level_val = "Intermediate"
        career_val = f"{selected_domain} Specialist / Engineer"

    st.subheader(f"📌 Domain Overview: {selected_domain}")

    ov_col1, ov_col2 = st.columns(2)
    ov_col1.metric("Recommended Entry Level", level_val)
    ov_col2.metric("Target Career Opportunity", career_val)

    st.markdown(
        f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 16px 20px; margin: 16px 0;">
            <span style="font-size: 0.8rem; text-transform: uppercase; color: #818cf8; font-weight: 700; letter-spacing: 0.05em;">Domain Overview</span>
            <p style="margin: 6px 0 0 0; color: #e2e8f0; font-size: 0.96rem; line-height: 1.6;">
                {desc}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Requirement 3: Recommended Roadmap for EVERY Domain
    st.subheader(f"🗺️ Recommended Roadmap for {selected_domain}")

    roadmap_modules, is_auto = get_domain_roadmap(selected_branch, selected_domain)

    if is_auto:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; border-radius: 6px; padding: 12px 16px; margin: 12px 0 18px 0;">
                <span style="color: #fbbf24; font-weight: 700;">Auto-generated roadmap</span>
                <p style="margin: 4px 0 0 0; color: #e2e8f0; font-size: 0.9rem;">
                    This roadmap is auto-generated using standardized engineering milestones (Fundamentals, Core Concepts, Tools, Applied, Advanced, Capstone).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; border-radius: 6px; padding: 10px 16px; margin: 12px 0 18px 0;">
                <span style="color: #4ade80; font-weight: 700;">✅ Curated Curriculum Roadmap</span>
                <span style="color: #e2e8f0; font-size: 0.9rem; margin-left: 8px;">
                    Officially cataloged engineering track with validated lab milestones.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Show Roadmap Modules
    for idx, mod in enumerate(roadmap_modules):
        mod_topic, mod_level, est_hrs, project, resource_url = mod
        with st.expander(f"Module {idx + 1}: {mod_topic} • [{mod_level}] - {est_hrs} Hours"):
            st.write(f"**🎯 Practical Project:** {project}")
            st.write(f"**⏱️ Estimated Duration:** {est_hrs} learning hours")
            st.link_button("🌐 Open Resource Link", resource_url)

    st.divider()

    # Domain Resources (Using 4-Level Fallback Engine)
    st.subheader(f"📚 Domain Learning Resources & Courses")
    rec_courses, fb_level, fb_msg = get_domain_courses(selected_domain, level_val, selected_branch)

    for c in rec_courses:
        with st.container(border=True):
            r_col1, r_col2 = st.columns([3, 1])
            with r_col1:
                st.markdown(f"#### {c['course']}")
                st.markdown(f"**Provider:** `{c['provider']}` &nbsp;|&nbsp; **Level:** `{c['level']}` &nbsp;|&nbsp; **Cost:** `{c['cost']}`")
            with r_col2:
                st.link_button("Start Learning ↗", c["resource_url"], use_container_width=True)


# ======================================================================================
# 10. PAGE 4: 📚 COURSE & RESOURCE RECOMMENDATIONS
# ======================================================================================

elif page == "📚 Course & Resource Recommendations":

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                Curated Learning Index
            </div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
                📚 Course & Resource Recommendations
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.98rem;">
                Explore verified courses, laboratory modules, and accredited engineering learning platforms with 4-level intelligent fallback.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_branches = get_all_branches()

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_branch = st.selectbox(
            "Select Engineering Branch",
            options=all_branches,
            index=0
        )

    branch_domains = get_branch_domains(selected_branch)

    with col2:
        selected_domain = st.selectbox(
            "Select Technical Domain",
            options=branch_domains,
            index=0
        )

    with col3:
        selected_level = st.selectbox(
            "Select Student Skill Level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=0
        )

    st.divider()

    # Requirement 4: Execute 4-Level Fallback Engine
    courses, fallback_level, explanation_msg = get_domain_courses(
        domain=selected_domain,
        level=selected_level,
        branch=selected_branch
    )

    st.subheader("🎯 Recommended Learning Resources")

    # Fallback explanation banner
    if fallback_level == 1:
        st.success(f"🎯 **Exact Level Match:** {explanation_msg}")
    elif fallback_level == 2:
        st.info(f"ℹ️ {explanation_msg}")
    elif fallback_level == 3:
        st.warning(f"📌 {explanation_msg}")
    else:
        st.info(f"🌐 {explanation_msg}")

    # Render Modern Resource Cards
    for idx, course in enumerate(courses):
        with st.container(border=True):
            c_col1, c_col2 = st.columns([3, 1])

            with c_col1:
                st.markdown(f"### {course['course']}")
                st.markdown(
                    f"**🏛️ Provider:** `{course['provider']}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**📊 Level:** `{course['level']}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**💰 Cost:** `{course['cost']}`"
                )

            with c_col2:
                st.write("")
                st.link_button("Start Learning ↗", course["resource_url"], use_container_width=True)

    # Actionable Engineering Study Strategy
    st.divider()
    st.subheader("💡 Learning Strategy & Certification Guidance")

    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #a5b4fc;">🚀 How to maximize these engineering resources:</h4>
            <ol style="margin: 0 0 0 20px; padding: 0; color: #cbd5e1; font-size: 0.94rem; line-height: 1.7;">
                <li><strong>Hands-on Implementation:</strong> Follow every theoretical lecture with code or CAD modeling immediately.</li>
                <li><strong>Version Control & Documentation:</strong> Commit every mini-project to GitHub with a clean README outlining your design decisions.</li>
                <li><strong>Capstone Integration:</strong> Combine 2 to 3 modular exercises into a comprehensive final-year project artifact.</li>
                <li><strong>Peer Review:</strong> Present your prototypes during group study or technical club sessions to practice technical communication.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
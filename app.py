import os
import json
import time
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
import gradio as gr

from data_loader import MAJORS_DATA, CLUBS_DATA, get_majors_summary, get_clubs_summary

# ==========================================
# 1. PYDANTIC SCHEMAS FOR AGENT CHAIN
# ==========================================

class StudentProfile(BaseModel):
    primary_focus: str = Field(description="Main academic or personal focus area")
    secondary_interests: List[str] = Field(description="List of secondary hobbies or topics")
    core_values: List[str] = Field(description="Key personal values or drivers (e.g. innovation, service, leadership)")
    career_aspirations: str = Field(description="Long-term career or dream project goal")

class MajorMatch(BaseModel):
    major_name: str
    category: str
    match_score: int
    rationale: str
    recommended_courses: List[str]

class ClubMatch(BaseModel):
    club_name: str
    rationale: str
    recommended_role: str

class CampusRoadmap(BaseModel):
    welcome_message: str
    top_majors: List[MajorMatch]
    top_clubs: List[ClubMatch]
    first_year_focus: str
    second_year_focus: str
    third_year_focus: str
    fourth_year_focus: str
    admissions_next_steps: List[str]

# ==========================================
# 2. AGENTIC CHAIN WORKFLOW LOGIC
# ==========================================

def run_agentic_pipeline(student_raw_input: str) -> Tuple[str, str, str, str, str]:
    """
    Executes the 4-step Agentic Chain Workflow:
    Step 1: Interest Extractor Agent -> StudentProfile
    Step 2: Major Matcher Agent -> List[MajorMatch] (Filtered from majors.html)
    Step 3: Club Matcher Agent -> List[ClubMatch] (Filtered from clubs.html)
    Step 4: Roadmap Synthesizer Agent -> CampusRoadmap (Markdown report)
    """
    if not student_raw_input or len(student_raw_input.strip()) < 5:
        empty_msg = "⚠️ Please provide a few sentences about your interests, hobbies, or career goals."
        return empty_msg, "{}", "[]", "[]", "Log trace idle."

    start_time = time.time()
    
    # ----------------------------------------------------
    # AGENT 1: Interest Extractor
    # ----------------------------------------------------
    profile = extract_student_profile(student_raw_input)
    agent1_json = json.dumps(profile.dict(), indent=2)
    
    # ----------------------------------------------------
    # AGENT 2 & 3: Parallel Matchers (Majors & Clubs)
    # ----------------------------------------------------
    matched_majors = match_majors(profile, MAJORS_DATA)
    agent2_json = json.dumps([m.dict() for m in matched_majors], indent=2)
    
    matched_clubs = match_clubs(profile, CLUBS_DATA)
    agent3_json = json.dumps([c.dict() for c in matched_clubs], indent=2)
    
    # ----------------------------------------------------
    # AGENT 4: Roadmap Synthesizer
    # ----------------------------------------------------
    roadmap = synthesize_roadmap(profile, matched_majors, matched_clubs)
    
    # Generate formatted Markdown output for the main student view
    markdown_output = render_roadmap_markdown(roadmap)
    
    elapsed = round(time.time() - start_time, 2)
    telemetry_log = f"✅ Agentic Chain Executed Successfully in {elapsed}s\n" \
                    f"- Extractor Agent: Done (1 Profile created)\n" \
                    f"- Major Matcher Agent: Done ({len(matched_majors)} majors filtered out of {len(MAJORS_DATA)} available)\n" \
                    f"- Club Matcher Agent: Done ({len(matched_clubs)} clubs matched out of {len(CLUBS_DATA)} available)\n" \
                    f"- Synthesizer Agent: Done (4-Year Experience Preview generated)"

    return markdown_output, agent1_json, agent2_json, agent3_json, telemetry_log


def extract_student_profile(raw_input: str) -> StudentProfile:
    """Agent 1: Extracts structured attributes from student prose."""
    text_lower = raw_input.lower()
    
    # Domain keyword detection
    interests = []
    if any(k in text_lower for k in ["code", "computer", "ai", "robot", "software", "tech", "game"]):
        interests.append("Artificial Intelligence & Technology")
    if any(k in text_lower for k in ["med", "health", "bio", "doctor", "biology", "hospital", "patient"]):
        interests.append("Healthcare & Medicine")
    if any(k in text_lower for k in ["business", "startup", "company", "finance", "market", "consulting"]):
        interests.append("Entrepreneurship & Business Strategy")
    if any(k in text_lower for k in ["art", "design", "music", "sing", "film", "drama", "creative"]):
        interests.append("Creative Arts & Media")
    if any(k in text_lower for k in ["climate", "environment", "nature", "sustain", "eco"]):
        interests.append("Environmental Action & Sustainability")

    if not interests:
        interests = ["Interdisciplinary Exploration", "Leadership"]
        
    values = ["Innovation", "Collaboration", "Real-World Impact"]
    
    return StudentProfile(
        primary_focus=interests[0],
        secondary_interests=interests[1:] if len(interests) > 1 else ["Community Engagement"],
        core_values=values,
        career_aspirations=f"Aspiring leader in {interests[0]} with a passion for driving innovation."
    )


def match_majors(profile: StudentProfile, majors: List[Dict[str, str]]) -> List[MajorMatch]:
    """Agent 2: Scores and selects top matching majors from dataset."""
    focus = profile.primary_focus.lower()
    matched = []
    
    # Score candidates
    for m in majors:
        name = m["name"]
        cat = m["category"]
        name_lower = name.lower()
        cat_lower = cat.lower()
        
        score = 60
        if "tech" in focus and ("computer" in name_lower or "engineering" in name_lower or "physics" in name_lower):
            score += 35
        elif "health" in focus and ("bio" in name_lower or "health" in name_lower or "chem" in name_lower):
            score += 35
        elif "business" in focus and ("business" in name_lower or "account" in name_lower or "bba" in name_lower):
            score += 35
        elif "arts" in focus and ("art" in name_lower or "music" in name_lower or "film" in name_lower):
            score += 35
        elif cat_lower in focus:
            score += 25
            
        if score > 75:
            matched.append(MajorMatch(
                major_name=name,
                category=cat,
                match_score=min(score, 98),
                rationale=f"Directly aligns with your interest in {profile.primary_focus} and hands-on career goals.",
                recommended_courses=[f"Introduction to {name}", f"Advanced Seminar in {cat}"]
            ))
            
    # Fallback to defaults if specific keyword matching is sparse
    if not matched:
        matched = [
            MajorMatch(
                major_name=majors[0]["name"],
                category=majors[0]["category"],
                match_score=92,
                rationale="Comprehensive foundational program with flexible elective tracks.",
                recommended_courses=["Core Academic Seminar", "Specialized Field Research"]
            ),
            MajorMatch(
                major_name=majors[1]["name"] if len(majors) > 1 else "BS in Business Analytics",
                category=majors[1]["category"] if len(majors) > 1 else "Business & Economics",
                match_score=87,
                rationale="Offers strong cross-disciplinary skills valued by recruiters.",
                recommended_courses=["Applied Analytics", "Capstone Leadership Project"]
            )
        ]
        
    return matched[:3]


def match_clubs(profile: StudentProfile, clubs: List[Dict[str, str]]) -> List[ClubMatch]:
    """Agent 3: Scores and selects top matching student clubs from dataset."""
    focus = profile.primary_focus.lower()
    matched = []
    
    for c in clubs:
        name = c["name"]
        desc = c["description"]
        combined = (name + " " + desc).lower()
        
        score = 50
        if "tech" in focus and ("ai" in combined or "robot" in combined or "consulting" in combined or "tech" in combined):
            score += 40
        elif "health" in focus and ("med" in combined or "mind" in combined or "health" in combined):
            score += 40
        elif "business" in focus and ("business" in name.lower() or "consulting" in combined or "account" in combined or "alpfa" in combined):
            score += 40
        elif "arts" in focus and ("capella" in combined or "art" in combined or "music" in combined):
            score += 40
            
        if score >= 80:
            matched.append(ClubMatch(
                club_name=name,
                rationale=f"{desc[:110]}...",
                recommended_role="Project Team Member / Committee Chair"
            ))
            
    if not matched:
        # Default top clubs from dataset
        matched = [
            ClubMatch(club_name=clubs[0]["name"], rationale=clubs[0]["description"][:110] + "...", recommended_role="Student Consultant"),
            ClubMatch(club_name=clubs[3]["name"] if len(clubs) > 3 else "Hofstra Student Government", rationale="Develop leadership and engage with campus events.", recommended_role="Active Representative")
        ]
        
    return matched[:3]


def synthesize_roadmap(profile: StudentProfile, majors: List[MajorMatch], clubs: List[ClubMatch]) -> CampusRoadmap:
    """Agent 4: Combines outputs into a unified 4-year campus roadmap."""
    top_major_names = ", ".join([m.major_name for m in majors])
    top_club_names = ", ".join([c.club_name for c in clubs])
    
    return CampusRoadmap(
        welcome_message=f"Welcome to Hofstra University! Based on your interest in **{profile.primary_focus}**, our agentic counseling chain has curated a personalized academic & extracurricular roadmap for you.",
        top_majors=majors,
        top_clubs=clubs,
        first_year_focus=f"Explore foundational courses in **{majors[0].major_name}**, attend common hour meetings for **{clubs[0].club_name}**, and connect with your faculty mentor.",
        second_year_focus=f"Take on leadership projects in **{clubs[0].club_name}**, declare your concentration, and apply for undergraduate research or campus lab assistantships.",
        third_year_focus=f"Participate in Hofstra's Internship & Career Fair, complete an industry co-op related to **{profile.primary_focus}**, and consider study abroad or NYC networking events.",
        fourth_year_focus=f"Complete your Senior Capstone Project in **{majors[0].major_name}**, serve as an executive board member in student organizations, and prepare for graduation or graduate school admissions.",
        admissions_next_steps=[
            "📅 **Schedule a Campus Tour**: Experience Hofstra's state-of-the-art labs and student center in person.",
            "💬 **Connect with a Faculty Advisor**: Meet professors in your matched program during today's Academic Fair.",
            "📝 **Submit Your Application**: Apply via the Common App or Hofstra Application prior to Early Action deadline.",
            "🎁 **Explore Merit Scholarships**: Financial aid counselors are available today in the Admission Center."
        ]
    )


def render_roadmap_markdown(roadmap: CampusRoadmap) -> str:
    """Renders the CampusRoadmap object as rich GitHub Markdown for display."""
    md = []
    md.append(f"### 🌟 {roadmap.welcome_message}\n")
    
    md.append("---")
    md.append("### 📚 Top Matched Undergraduate Programs")
    for m in roadmap.top_majors:
        md.append(f"#### 🎓 **{m.major_name}** *(Match Score: {m.match_score}%)*")
        md.append(f"- **Category**: {m.category}")
        md.append(f"- **Why It Fits**: {m.rationale}")
        md.append(f"- **Recommended Sample Courses**: `{', '.join(m.recommended_courses)}`\n")
        
    md.append("---")
    md.append("### 🏆 Recommended Campus Clubs & Student Organizations")
    for c in roadmap.top_clubs:
        md.append(f"#### 🤝 **{c.club_name}**")
        md.append(f"- **About & Impact**: {c.rationale}")
        md.append(f"- **Suggested Entry Role**: `{c.recommended_role}`\n")
        
    md.append("---")
    md.append("### 🗺️ Your 4-Year Experience Preview")
    md.append(f"- **Year 1 (Foundation & Discovery)**: {roadmap.first_year_focus}")
    md.append(f"- **Year 2 (Leadership & Deep Dive)**: {roadmap.second_year_focus}")
    md.append(f"- **Year 3 (Internships & Real-World Impact)**: {roadmap.third_year_focus}")
    md.append(f"- **Year 4 (Senior Capstone & Career Launch)**: {roadmap.fourth_year_focus}\n")
    
    md.append("---")
    md.append("### 📌 Admission Event Next Steps")
    for step in roadmap.admissions_next_steps:
        md.append(f"- {step}")
        
    return "\n".join(md)


# ==========================================
# 3. GRADIO USER INTERFACE
# ==========================================

custom_css = """
.container { max-width: 1100px; margin: 0 auto; }
.header-box {
    background: linear-gradient(135deg, #003366 0%, #0055a5 50%, #001f3f 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}
.header-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; }
.header-box p { margin-top: 8px; font-size: 1.1rem; color: #e0e8f5; }
.badge {
    background-color: #f39c12;
    color: #111;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: bold;
    display: inline-block;
    margin-top: 10px;
}
"""

def create_gradio_app():
    with gr.Blocks(title="Smart Major & Campus Life Matcher", css=custom_css) as app:
        
        # Header Banner
        gr.HTML("""
        <div class="header-box">
            <h1>🎓 Hofstra University — Smart Major & Campus Life Matcher</h1>
            <p>Multi-Agent Chain Workflow Demo for Admissions & Open House Events</p>
            <div class="badge">Powered by Agentic Routing & Data Extraction (majors.html & clubs.html)</div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 📝 Tell Us About Yourself")
                student_input = gr.Textbox(
                    label="What are your interests, hobbies, favorite classes, or career dreams?",
                    placeholder="e.g. I love coding games, building electronics, playing guitar, and I want to start my own robotics or AI tech company...",
                    lines=4
                )
                
                gr.Markdown("#### 💡 Quick Presets for Demo Testing:")
                with gr.Row():
                    btn_tech = gr.Button("🚀 Tech & AI Innovator")
                    btn_med = gr.Button("🩺 Pre-Med & Healthcare")
                    btn_biz = gr.Button("💼 Entrepreneurship & Business")
                    btn_arts = gr.Button("🎨 Music & Digital Media")
                    
                submit_btn = gr.Button("✨ Run Agentic Matching Chain", variant="primary", size="lg")
                
            with gr.Column(scale=7):
                gr.Markdown("### 🎯 Agentic Workflow Results")
                
                with gr.Tabs():
                    with gr.TabItem("🗺️ 4-Year Campus Preview"):
                        output_markdown = gr.Markdown("*(Run the agentic chain to see your personalized 4-Year Hofstra Experience preview)*")
                        
                    with gr.TabItem("🔍 Agent 1 Trace (Profile Extractor)"):
                        output_agent1 = gr.Code(label="Extracted StudentProfile JSON", language="json")
                        
                    with gr.TabItem("📚 Agent 2 Trace (Major Matcher)"):
                        output_agent2 = gr.Code(label="Matched Majors from majors.html", language="json")
                        
                    with gr.TabItem("🏆 Agent 3 Trace (Club Matcher)"):
                        output_agent3 = gr.Code(label="Matched Student Clubs from clubs.html", language="json")
                        
                    with gr.TabItem("⚙️ System Telemetry"):
                        output_telemetry = gr.Textbox(label="Agent Chain Execution Telemetry", lines=6)

        # Preset click listeners
        btn_tech.click(
            fn=lambda: "I love robotics, python programming, building microcontrollers, and want to study artificial intelligence to start a software company.",
            outputs=student_input
        )
        btn_med.click(
            fn=lambda: "I am passionate about biology, volunteering at community health clinics, mental health awareness, and preparing for medical school.",
            outputs=student_input
        )
        btn_biz.click(
            fn=lambda: "I enjoy marketing, leading high school clubs, stock market trading, consulting small businesses, and social entrepreneurship.",
            outputs=student_input
        )
        btn_arts.click(
            fn=lambda: "I love graphic design, video editing, performing in choir, singing a-cappella, and studying digital media production.",
            outputs=student_input
        )

        # Submit listener
        submit_btn.click(
            fn=run_agentic_pipeline,
            inputs=[student_input],
            outputs=[output_markdown, output_agent1, output_agent2, output_agent3, output_telemetry]
        )
        
    return app

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

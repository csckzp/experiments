import os
import json
import time
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
import gradio as gr

# LangChain Imports
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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
    major_name: str = Field(description="Name of matched major from dataset")
    category: str = Field(description="Academic category/department")
    match_score: int = Field(description="Match confidence score between 1 and 100")
    rationale: str = Field(description="Why this major fits the student profile")
    recommended_courses: List[str] = Field(description="Sample course highlights")

class MajorMatchList(BaseModel):
    matches: List[MajorMatch] = Field(description="Top matched majors")

class ClubMatch(BaseModel):
    club_name: str = Field(description="Name of matched student organization")
    rationale: str = Field(description="Why this club fits the student profile")
    recommended_role: str = Field(description="Suggested entry involvement role")

class ClubMatchList(BaseModel):
    matches: List[ClubMatch] = Field(description="Top matched student clubs")

class CampusRoadmap(BaseModel):
    welcome_message: str = Field(description="Welcoming ambassador greeting")
    first_year_focus: str = Field(description="Year 1 foundation and discovery focus")
    second_year_focus: str = Field(description="Year 2 leadership and deep dive focus")
    third_year_focus: str = Field(description="Year 3 internship and real-world impact focus")
    fourth_year_focus: str = Field(description="Year 4 senior capstone and career launch focus")
    admissions_next_steps: List[str] = Field(description="Actionable admissions next steps for Open House attendees")

# ==========================================
# 2. LANGCHAIN AGENT FACTORY (create_agent)
# ==========================================

def create_agent(llm, system_prompt: str, pydantic_schema=None):
    """
    Factory function creating an LLM Agent chain using LangChain ChatPromptTemplate,
    ChatOllama model, and optional PydanticOutputParser for structured JSON outputs.
    """
    if pydantic_schema and LANGCHAIN_AVAILABLE:
        parser = PydanticOutputParser(pydantic_object=pydantic_schema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nCRITICAL: You MUST respond strictly in valid JSON matching this format instructions:\n{format_instructions}"),
            ("user", "{input_text}")
        ]).partial(format_instructions=parser.get_format_instructions())
        chain = prompt | llm | parser
    elif LANGCHAIN_AVAILABLE:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input_text}")
        ])
        chain = prompt | llm
    else:
        chain = None
    return chain

# ==========================================
# 3. MULTI-AGENT PIPELINE RUNNER
# ==========================================

def run_agentic_pipeline(
    student_raw_input: str,
    model_name: str = "gemma4:e2b",
    base_url: str = "http://localhost:11434"
) -> Tuple[str, str, str, str, str]:
    """
    Executes the 4-Agent LangChain Pipeline:
    Agent 1 (Extractor Agent) -> Agent 2 (Major Matcher) & Agent 3 (Club Matcher) -> Agent 4 (Roadmap Synthesizer)
    """
    if not student_raw_input or len(student_raw_input.strip()) < 5:
        empty_msg = "⚠️ Please provide a few sentences about your interests, hobbies, or career goals."
        return empty_msg, "{}", "[]", "[]", "Log trace idle."

    start_time = time.time()
    telemetry_logs = []
    telemetry_logs.append(f"⚙️ Target LLM Model: '{model_name}' @ '{base_url}'")

    # Initialize LangChain ChatOllama LLM Instance
    llm = None
    if LANGCHAIN_AVAILABLE:
        try:
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.2
            )
            telemetry_logs.append("✅ LangChain & ChatOllama Client initialized successfully.")
        except Exception as e:
            telemetry_logs.append(f"⚠️ ChatOllama initialization notice: {e}")

    # ----------------------------------------------------
    # AGENT 1: Interest Extractor Agent
    # ----------------------------------------------------
    telemetry_logs.append("🚀 [Agent 1] Invoking Extractor Agent...")
    extractor_prompt = (
        "You are an Admissions Counselor Agent specializing in analyzing student interests, hobbies, values, "
        "and career aspirations from raw student descriptions."
    )
    extractor_agent = create_agent(llm, extractor_prompt, pydantic_schema=StudentProfile)
    
    profile: Optional[StudentProfile] = None
    if extractor_agent and llm:
        try:
            profile = extractor_agent.invoke({"input_text": student_raw_input})
            telemetry_logs.append("✅ [Agent 1] Extractor Agent successfully generated StudentProfile via Ollama LLM.")
        except Exception as err:
            telemetry_logs.append(f"⚠️ [Agent 1] LLM call fell back to heuristic parser ({err})")
            profile = fallback_extract_profile(student_raw_input)
    else:
        profile = fallback_extract_profile(student_raw_input)
        telemetry_logs.append("ℹ️ [Agent 1] Using heuristic parser (Ollama offline/not reachable).")

    agent1_json = json.dumps(profile.dict(), indent=2)

    # ----------------------------------------------------
    # AGENT 2: Major Matcher Agent
    # ----------------------------------------------------
    telemetry_logs.append("🚀 [Agent 2] Invoking Major Matcher Agent against majors.html dataset...")
    majors_context = get_majors_summary(limit=40)
    major_prompt = (
        "You are an Academic Program Matching Agent. Given the student profile, analyze the available university majors "
        "below and select the top 2 to 3 most relevant majors with rationale and sample course highlights.\n\n"
        f"Available Majors Dataset:\n{majors_context}"
    )
    major_agent = create_agent(llm, major_prompt, pydantic_schema=MajorMatchList)
    
    matched_majors: List[MajorMatch] = []
    if major_agent and llm:
        try:
            result = major_agent.invoke({"input_text": agent1_json})
            matched_majors = result.matches
            telemetry_logs.append(f"✅ [Agent 2] Major Matcher Agent selected {len(matched_majors)} majors via LLM.")
        except Exception as err:
            telemetry_logs.append(f"⚠️ [Agent 2] LLM call fell back to dataset filter ({err})")
            matched_majors = fallback_match_majors(profile, MAJORS_DATA)
    else:
        matched_majors = fallback_match_majors(profile, MAJORS_DATA)
        telemetry_logs.append(f"ℹ️ [Agent 2] Matched {len(matched_majors)} majors from dataset.")

    agent2_json = json.dumps([m.dict() for m in matched_majors], indent=2)

    # ----------------------------------------------------
    # AGENT 3: Club Matcher Agent
    # ----------------------------------------------------
    telemetry_logs.append("🚀 [Agent 3] Invoking Club Matcher Agent against clubs.html dataset...")
    clubs_context = get_clubs_summary(limit=40)
    club_prompt = (
        "You are a Campus Life & Student Organization Matching Agent. Given the student profile, analyze the available clubs "
        "below and select the top 2 to 3 most engaging campus organizations with rationale and suggested student role.\n\n"
        f"Available Student Clubs Dataset:\n{clubs_context}"
    )
    club_agent = create_agent(llm, club_prompt, pydantic_schema=ClubMatchList)
    
    matched_clubs: List[ClubMatch] = []
    if club_agent and llm:
        try:
            result = club_agent.invoke({"input_text": agent1_json})
            matched_clubs = result.matches
            telemetry_logs.append(f"✅ [Agent 3] Club Matcher Agent selected {len(matched_clubs)} clubs via LLM.")
        except Exception as err:
            telemetry_logs.append(f"⚠️ [Agent 3] LLM call fell back to dataset filter ({err})")
            matched_clubs = fallback_match_clubs(profile, CLUBS_DATA)
    else:
        matched_clubs = fallback_match_clubs(profile, CLUBS_DATA)
        telemetry_logs.append(f"ℹ️ [Agent 3] Matched {len(matched_clubs)} campus clubs from dataset.")

    agent3_json = json.dumps([c.dict() for c in matched_clubs], indent=2)

    # ----------------------------------------------------
    # AGENT 4: Roadmap Synthesizer Agent
    # ----------------------------------------------------
    telemetry_logs.append("🚀 [Agent 4] Invoking Roadmap Synthesizer Agent...")
    synthesizer_prompt = (
        "You are an Admissions Ambassador Synthesizer Agent. Combine the student profile, matched majors, and matched clubs "
        "into a comprehensive 4-year campus experience preview and actionable open house next steps."
    )
    synthesizer_agent = create_agent(llm, synthesizer_prompt, pydantic_schema=CampusRoadmap)
    
    roadmap: Optional[CampusRoadmap] = None
    synth_input = json.dumps({
        "profile": profile.dict(),
        "majors": [m.dict() for m in matched_majors],
        "clubs": [c.dict() for c in matched_clubs]
    }, indent=2)
    
    if synthesizer_agent and llm:
        try:
            roadmap = synthesizer_agent.invoke({"input_text": synth_input})
            telemetry_logs.append("✅ [Agent 4] Roadmap Synthesizer Agent generated 4-Year Preview via LLM.")
        except Exception as err:
            telemetry_logs.append(f"⚠️ [Agent 4] LLM call fell back to synthesizer ({err})")
            roadmap = fallback_synthesize_roadmap(profile, matched_majors, matched_clubs)
    else:
        roadmap = fallback_synthesize_roadmap(profile, matched_majors, matched_clubs)
        telemetry_logs.append("ℹ️ [Agent 4] Synthesized 4-Year Experience Preview.")

    elapsed = round(time.time() - start_time, 2)
    telemetry_logs.append(f"⏱️ Total Agentic Pipeline Time: {elapsed}s")
    
    markdown_output = render_roadmap_markdown(roadmap, matched_majors, matched_clubs)
    return markdown_output, agent1_json, agent2_json, agent3_json, "\n".join(telemetry_logs)


# ==========================================
# 4. FALLBACK HELPERS (Ensure zero-crash operation)
# ==========================================

def fallback_extract_profile(raw_input: str) -> StudentProfile:
    text_lower = raw_input.lower()
    interests = []
    if any(k in text_lower for k in ["code", "computer", "ai", "robot", "software", "tech", "game"]):
        interests.append("Artificial Intelligence & Technology")
    if any(k in text_lower for k in ["med", "health", "bio", "doctor", "biology"]):
        interests.append("Healthcare & Pre-Medical Sciences")
    if any(k in text_lower for k in ["business", "startup", "company", "finance", "market"]):
        interests.append("Entrepreneurship & Business Innovation")
    if any(k in text_lower for k in ["art", "design", "music", "sing", "film", "creative"]):
        interests.append("Creative Arts & Digital Media")
        
    if not interests:
        interests = ["Interdisciplinary Exploration", "Leadership"]
        
    return StudentProfile(
        primary_focus=interests[0],
        secondary_interests=interests[1:] if len(interests) > 1 else ["Community Leadership"],
        core_values=["Innovation", "Collaboration", "Global Impact"],
        career_aspirations=f"Aspiring leader in {interests[0]} with a drive for real-world impact."
    )

def fallback_match_majors(profile: StudentProfile, majors: List[Dict[str, str]]) -> List[MajorMatch]:
    focus = profile.primary_focus.lower()
    matched = []
    for m in majors:
        name, cat = m["name"], m["category"]
        score = 65
        if "tech" in focus and any(k in name.lower() for k in ["computer", "engineering", "physics", "cyber"]):
            score += 30
        elif "health" in focus and any(k in name.lower() for k in ["bio", "health", "chem"]):
            score += 30
        elif "business" in focus and any(k in name.lower() for k in ["business", "account", "bba"]):
            score += 30
        elif "arts" in focus and any(k in name.lower() for k in ["art", "music", "film"]):
            score += 30
            
        if score > 75:
            matched.append(MajorMatch(
                major_name=name,
                category=cat,
                match_score=min(score, 98),
                rationale=f"Directly aligns with your primary focus in {profile.primary_focus}.",
                recommended_courses=[f"Intro to {name}", f"Advanced {cat} Project"]
            ))
            
    return matched[:3] if matched else [
        MajorMatch(major_name=majors[0]["name"], category=majors[0]["category"], match_score=92, rationale="Top foundational degree program.", recommended_courses=["Core Seminar", "Capstone Research"])
    ]

def fallback_match_clubs(profile: StudentProfile, clubs: List[Dict[str, str]]) -> List[ClubMatch]:
    focus = profile.primary_focus.lower()
    matched = []
    for c in clubs:
        combined = (c["name"] + " " + c["description"]).lower()
        if any(k in combined for k in ["ai", "robot", "tech", "consulting", "med", "art", "music", "business"]):
            matched.append(ClubMatch(
                club_name=c["name"],
                rationale=c["description"][:110] + "...",
                recommended_role="Active Team Member / Officer"
            ))
    return matched[:3] if matched else [
        ClubMatch(club_name=clubs[0]["name"], rationale=clubs[0]["description"][:110] + "...", recommended_role="Student Consultant")
    ]

def fallback_synthesize_roadmap(profile: StudentProfile, majors: List[MajorMatch], clubs: List[ClubMatch]) -> CampusRoadmap:
    return CampusRoadmap(
        welcome_message=f"Welcome to Hofstra University! Based on your focus in **{profile.primary_focus}**, our LangChain agent pipeline has synthesized your personalized 4-year campus roadmap.",
        first_year_focus=f"Take core foundational courses in **{majors[0].major_name if majors else 'your major'}** and join **{clubs[0].club_name if clubs else 'campus clubs'}** during Fall Common Hour.",
        second_year_focus=f"Engage in undergraduate research, run for committee leadership roles, and connect with faculty mentors.",
        third_year_focus=f"Participate in the Hofstra Career & Internship Fair, complete industry co-ops, and attend NYC networking forums.",
        fourth_year_focus=f"Complete your Senior Capstone Project, lead student organization initiatives, and prepare for graduate launch.",
        admissions_next_steps=[
            "📅 **Schedule a Campus Tour**: Explore our laboratories, library, and student center.",
            "💬 **Meet Faculty Members**: Talk with department professors during today's Academic Fair.",
            "📝 **Submit Your Application**: Complete the Common App prior to the Early Action deadline.",
            "🎁 **Consult Financial Aid**: Visit the Admission Center for merit scholarship guidance."
        ]
    )

def render_roadmap_markdown(roadmap: CampusRoadmap, majors: List[MajorMatch], clubs: List[ClubMatch]) -> str:
    md = []
    md.append(f"### 🌟 {roadmap.welcome_message}\n")
    md.append("---")
    md.append("### 📚 Top Matched Undergraduate Programs (from majors.html)")
    for m in majors:
        md.append(f"#### 🎓 **{m.major_name}** *(Match Score: {m.match_score}%)*")
        md.append(f"- **Category**: {m.category}")
        md.append(f"- **Why It Fits**: {m.rationale}")
        md.append(f"- **Sample Course Highlights**: `{', '.join(m.recommended_courses)}`\n")
        
    md.append("---")
    md.append("### 🏆 Recommended Student Organizations (from clubs.html)")
    for c in clubs:
        md.append(f"#### 🤝 **{c.club_name}**")
        md.append(f"- **Impact & Focus**: {c.rationale}")
        md.append(f"- **Suggested Role**: `{c.recommended_role}`\n")
        
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
# 5. GRADIO USER INTERFACE
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
    with gr.Blocks(title="Smart Major & Campus Life Matcher (LangChain + Ollama)", css=custom_css) as app:
        
        # Header Banner
        gr.HTML("""
        <div class="header-box">
            <h1>🎓 Hofstra University — Smart Major & Campus Life Matcher</h1>
            <p>LangChain Multi-Agent Architecture powered by Ollama (gemma4:e2b) & create_agent</p>
            <div class="badge">Live Extracted Data from majors.html & clubs.html</div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### ⚙️ LLM & Agent Config")
                with gr.Row():
                    model_input = gr.Textbox(label="Ollama Model", value="gemma4:e2b", placeholder="e.g. gemma4:e2b, gemma2:2b, llama3.1")
                    url_input = gr.Textbox(label="Ollama Base URL", value="http://localhost:11434")

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
                    
                submit_btn = gr.Button("✨ Run LangChain Agent Chain", variant="primary", size="lg")
                
            with gr.Column(scale=7):
                gr.Markdown("### 🎯 Agentic Workflow Results")
                
                with gr.Tabs():
                    with gr.TabItem("🗺️ 4-Year Campus Preview"):
                        output_markdown = gr.Markdown("*(Run the agent chain to see your personalized 4-Year Hofstra Experience preview)*")
                        
                    with gr.TabItem("🔍 Agent 1 Trace (Extractor create_agent)"):
                        output_agent1 = gr.Code(label="Extracted StudentProfile JSON", language="json")
                        
                    with gr.TabItem("📚 Agent 2 Trace (Major Matcher)"):
                        output_agent2 = gr.Code(label="Matched Majors from majors.html", language="json")
                        
                    with gr.TabItem("🏆 Agent 3 Trace (Club Matcher)"):
                        output_agent3 = gr.Code(label="Matched Student Clubs from clubs.html", language="json")
                        
                    with gr.TabItem("⚙️ System Telemetry & Logs"):
                        output_telemetry = gr.Textbox(label="LangChain Execution Telemetry", lines=10)

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
            inputs=[student_input, model_input, url_input],
            outputs=[output_markdown, output_agent1, output_agent2, output_agent3, output_telemetry]
        )
        
    return app

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

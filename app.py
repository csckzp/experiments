import json
import time
from typing import List, Tuple
from pydantic import BaseModel, Field
import gradio as gr

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from data_loader import MAJORS_DATA, CLUBS_DATA, get_majors_summary, get_clubs_summary

# ==========================================
# 1. PYDANTIC SCHEMAS FOR AGENT PIPELINE
# ==========================================

class StudentProfile(BaseModel):
    primary_focus: str = Field(description="Main academic or personal focus area")
    secondary_interests: List[str] = Field(description="List of secondary hobbies or topics")
    core_values: List[str] = Field(description="Key personal values or drivers")
    career_aspirations: str = Field(description="Long-term career or project goal")

class MajorMatch(BaseModel):
    major_name: str = Field(description="Name of matched major from dataset")
    category: str = Field(description="Academic category/department")
    match_score: int = Field(description="Match confidence score (1-100)")
    rationale: str = Field(description="Why this major fits the student profile")
    recommended_courses: List[str] = Field(description="Sample course highlights")

class MajorMatchList(BaseModel):
    matches: List[MajorMatch] = Field(description="Top matched majors")

class ClubMatch(BaseModel):
    club_name: str = Field(description="Name of matched student organization")
    rationale: str = Field(description="Why this club fits the student profile")
    recommended_role: str = Field(description="Suggested student role")

class ClubMatchList(BaseModel):
    matches: List[ClubMatch] = Field(description="Top matched student clubs")

class CampusRoadmap(BaseModel):
    welcome_message: str = Field(description="Welcoming ambassador greeting")
    first_year_focus: str = Field(description="Year 1 foundation and discovery focus")
    second_year_focus: str = Field(description="Year 2 leadership and research focus")
    third_year_focus: str = Field(description="Year 3 internship and co-op focus")
    fourth_year_focus: str = Field(description="Year 4 senior capstone and career launch focus")
    admissions_next_steps: List[str] = Field(description="Actionable open house next steps")

# ==========================================
# 2. LANGCHAIN AGENT FACTORY (create_agent)
# ==========================================

def create_agent(llm: ChatOllama, system_prompt: str, pydantic_schema=None):
    """Factory function creating a LangChain Agent chain with Pydantic output parsing."""
    if pydantic_schema:
        parser = PydanticOutputParser(pydantic_object=pydantic_schema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nCRITICAL: Respond strictly in valid JSON matching schema:\n{format_instructions}"),
            ("user", "{input_text}")
        ]).partial(format_instructions=parser.get_format_instructions())
        return prompt | llm | parser
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input_text}")
        ])
        return prompt | llm

# ==========================================
# 3. MULTI-AGENT PIPELINE RUNNER
# ==========================================

def run_agentic_pipeline(
    student_raw_input: str,
    model_name: str = "gemma4:e2b",
    base_url: str = "http://localhost:11434"
) -> Tuple[str, str, str, str, str]:
    if not student_raw_input.strip():
        return "⚠️ Please enter your interests.", "{}", "[]", "[]", "Log trace idle."

    start_time = time.time()
    
    # Initialize ChatOllama LLM
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.2)
    
    # --- AGENT 1: Interest Extractor Agent ---
    extractor_agent = create_agent(
        llm,
        "You are an Admissions Counselor Agent. Extract structured attributes from raw student prose.",
        pydantic_schema=StudentProfile
    )
    profile: StudentProfile = extractor_agent.invoke({"input_text": student_raw_input})
    agent1_json = json.dumps(profile.dict(), indent=2)
    
    # --- AGENT 2: Major Matcher Agent ---
    majors_context = get_majors_summary(limit=40)
    major_agent = create_agent(
        llm,
        f"You are an Academic Program Matching Agent. Select top 2-3 majors from dataset:\n{majors_context}",
        pydantic_schema=MajorMatchList
    )
    matched_majors_res: MajorMatchList = major_agent.invoke({"input_text": agent1_json})
    matched_majors = matched_majors_res.matches
    agent2_json = json.dumps([m.dict() for m in matched_majors], indent=2)
    
    # --- AGENT 3: Club Matcher Agent ---
    clubs_context = get_clubs_summary(limit=40)
    club_agent = create_agent(
        llm,
        f"You are a Campus Life Matching Agent. Select top 2-3 student clubs from dataset:\n{clubs_context}",
        pydantic_schema=ClubMatchList
    )
    matched_clubs_res: ClubMatchList = club_agent.invoke({"input_text": agent1_json})
    matched_clubs = matched_clubs_res.matches
    agent3_json = json.dumps([c.dict() for c in matched_clubs], indent=2)
    
    # --- AGENT 4: Roadmap Synthesizer Agent ---
    synthesizer_agent = create_agent(
        llm,
        "You are an Admissions Roadmap Synthesizer Agent. Combine inputs into a 4-year campus roadmap.",
        pydantic_schema=CampusRoadmap
    )
    synth_input = json.dumps({
        "profile": profile.dict(),
        "majors": [m.dict() for m in matched_majors],
        "clubs": [c.dict() for c in matched_clubs]
    }, indent=2)
    roadmap: CampusRoadmap = synthesizer_agent.invoke({"input_text": synth_input})
    
    elapsed = round(time.time() - start_time, 2)
    telemetry = (
        f"✅ LangChain Agentic Chain Executed in {elapsed}s\n"
        f"- Target Model: {model_name} @ {base_url}\n"
        f"- Agent 1 (Extractor): Generated StudentProfile JSON\n"
        f"- Agent 2 (Major Matcher): Matched {len(matched_majors)} majors from majors.html\n"
        f"- Agent 3 (Club Matcher): Matched {len(matched_clubs)} campus clubs from clubs.html\n"
        f"- Agent 4 (Synthesizer): Generated CampusRoadmap"
    )
    
    markdown_output = render_roadmap_markdown(roadmap, matched_majors, matched_clubs)
    return markdown_output, agent1_json, agent2_json, agent3_json, telemetry


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
# 4. GRADIO USER INTERFACE
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
    with gr.Blocks(title="Smart Major & Campus Life Matcher (LangChain)", css=custom_css) as app:
        
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
                    model_input = gr.Textbox(label="Ollama Model", value="gemma4:e2b")
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

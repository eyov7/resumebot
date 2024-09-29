# crew.py
from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain.llms.base import LLM
from langchain.agents import Tool
import os
from utils import get_serper_api_key, get_groq_api_key

# Import LiteLLM
from litellm import completion

# Set Groq API key
groq_api_key = get_groq_api_key()
os.environ['GROQ_API_KEY'] = groq_api_key


# Custom LLM Class
from typing import Optional, List, Mapping, Any

class GroqLLM(LLM):
    model_name: str = "groq/llama3-8b-8192"
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                api_key=groq_api_key,
            )
            text = response['choices'][0]['message']['content']
            if stop:
                for s in stop:
                    text = text.split(s)[0]
            return text.strip()
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            return "An error occurred during inference."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "groq"

# Initialize custom LLM
llm = GroqLLM()

# Environment Variables
os.environ["SERPER_API_KEY"] = get_serper_api_key()

from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path="./resume.md")
semantic_search_resume = MDXSearchTool(mdx="./resume.md")

# Agent 1: Career Pathfinder
career_pathfinder = Agent(
    role="Career Pathfinder",
    goal="Guide new graduates in exploring suitable career paths based on their resume, skill set, and interests",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "As a Career Pathfinder, your extensive knowledge "
        "in career guidance, job market trends, and personal "
        "development makes you an invaluable resource for "
        "new graduates. You started your journey with a passion "
        "for helping individuals discover their potential "
        "and navigate the complexities of career planning. "
        "With a background in human resources and career "
        "coaching, you have honed the ability to analyze "
        "resumes, assess skill sets, and align personal "
        "interests with viable career paths. Utilizing "
        "advanced tools and personalized advice, you aim "
        "to empower graduates to make informed decisions "
        "and pursue fulfilling careers."
    ),
)

# Agent 1.1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job postings to help job applicants",
    tools=[scrape_tool, search_tool],
    verbose=True,
    llm=llm,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched. "
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)

# Agent 2.2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do incredible research on job applicants "
         "to help them stand out in the job market",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)

# Agent 2: Job Search Strategist
job_search_strategist = Agent(
    role="Job Search Strategist",
    goal="Assist graduates in finding relevant job openings, navigating online job boards, and building a strong online presence "
         "to help them stand out in the job market",
    tools=[scrape_tool, search_tool, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "As a Job Search Strategist, your expertise in job market dynamics "
        "and online job search techniques is unparalleled. Your journey started "
        "with a keen interest in digital tools and platforms that revolutionize "
        "the job hunting process. With a background in recruitment and career "
        "coaching, you have developed a knack for identifying the best job "
        "opportunities, optimizing online profiles, and streamlining the application "
        "process. Your mission is to empower graduates to navigate the complex "
        "landscape of online job boards and build a compelling online presence that "
        "stands out to potential employers."
    ),
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for AI and ML Graduates",
    goal="Help graduates create standout resumes that effectively highlight their skills, experiences, and qualifications to attract potential employers.",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "As a Resume Strategist, your expertise in crafting impactful resumes that catch the eye of recruiters is unmatched. "
        "Your journey began with a passion for storytelling and a keen understanding of the job market's evolving demands. "
        "With a background in human resources and professional writing, you have honed the ability to translate a graduate's "
        "experiences and skills into compelling resume content. Your mission is to empower graduates by providing them with "
        "the tools and knowledge to create resumes that stand out in the competitive job market. "
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    ),
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Prepare graduates for job interviews by providing tailored questions, answers, and talking points based on their resume and career aspirations.",
    tools=[read_resume, semantic_search_resume],
    verbose=True,
    llm=llm,
    backstory=(
        "As an Interview Preparer, your exceptional ability to equip graduates with the skills and confidence needed for successful interviews sets you apart. "
        "Your journey began with a fascination for effective communication and a desire to help individuals present their best selves. "
        "With a background in human resources, coaching, and behavioral psychology, you have developed a deep understanding of what employers seek during interviews. "
        "Your mission is to empower graduates by providing them with personalized interview preparation, ensuring they can articulate their experiences, skills, and aspirations compellingly. "
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    ),
)

# Task for Career Pathfinder Agent
career_pathfinder_task = Task(
    description=(
        "Analyze the graduate's resume, skill set, the LinkedIn ({linkedin_url}) URL, "
        "interest ({interest}) and career goals ({career_goals}) to "
        "explore potential career paths. Use tools to evaluate the "
        "graduate's qualifications, match them with suitable career "
        "options, and assess personal interests. Provide a comprehensive "
        "report detailing the most relevant career paths, including "
        "necessary qualifications, potential job titles, industry trends, "
        "and further skill development recommendations."
    ),
    expected_output=(
        "A detailed report outlining suitable career paths for the graduate, "
        "including relevant job titles, industry trends, and recommendations "
        "for further skill development."
    ),
    output_file="career_paths_report.md",
    agent=career_pathfinder,
)

# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URL and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

# Task for Job Search Strategist Agent
job_search_strategist_task = Task(
    description=(
        "Using the graduate's resume and career aspirations, assist them in finding "
        "relevant job openings. Navigate online job boards and use semantic search "
        "tools to identify job listings that match the graduate's skill set and interests. "
        "Additionally, provide guidance on building a strong online presence, including "
        "optimizing their LinkedIn profile and other professional networking sites."
    ),
    expected_output=(
        "A list of relevant job openings tailored to the graduate's skills and interests, "
        "along with actionable recommendations for building and optimizing their online presence."
    ),
    output_file="job_search_strategy.md",
    context=[career_pathfinder_task],
    agent=job_search_strategist,
)

# Task for Resume Strategist Agent
resume_strategist_task = Task(
    description=(
        "Using the graduate's current resume, skills, and experiences, create a standout resume that effectively highlights "
        "their qualifications to attract potential employers. Employ tools to analyze and enhance resume content, ensuring it "
        "aligns with industry standards and job market demands. Tailor the resume to emphasize key strengths and achievements, "
        "making it compelling and visually appealing."
    ),
    expected_output=(
        "A polished and compelling resume that effectively highlights the graduate's skills, experiences, and qualifications, "
        "making them stand out to potential employers."
    ),
    output_file="standout_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist,
)

# Task for Interview Preparer Agent
interview_preparer_task = Task(
    description=(
        "Using the graduate's resume and career aspirations, prepare them for job interviews by providing tailored questions, answers, "
        "and talking points. Analyze the resume to identify key experiences and skills to highlight during interviews. Generate questions "
        "that are likely to be asked based on the graduate's career field and aspirations, along with well-crafted answers and talking points "
        "to ensure the graduate presents their best self."
    ),
    expected_output=(
        "A comprehensive interview preparation document containing tailored interview questions, well-crafted answers, and key talking points "
        "based on the graduate's resume and career aspirations."
    ),
    context=[
        career_pathfinder_task,
        job_search_strategist_task,
        resume_strategist_task,
        research_task,
        profile_task,
    ],
    output_file="interview_prep_guide.md",
    agent=interview_preparer,
)

# Initialize the Crew
job_application_crew = Crew(
    agents=[
        career_pathfinder,
        researcher,
        profiler,
        job_search_strategist,
        resume_strategist,
        interview_preparer
    ],
    tasks=[
        career_pathfinder_task,
        research_task,
        profile_task,
        job_search_strategist_task,
        resume_strategist_task,
        interview_preparer_task
    ],
    verbose=True
)

career_counselling_inputs = {
    'job_posting_url': 'https://jobs.apple.com/en-us/details/200554363/machine-learning-ai-internships?team=STDNT',
    "linkedin_url": "https://www.linkedin.com/in/everolivares/",
    "github_url": "https://github.com/eyov7",
    "interest": "Artificial Intelligence, Machine Learning, Data Analysis & Visualization, Data Science",
    "career_goals": """As an AI Researcher: Strive to stay at the forefront of research trends and technologies, continually seeking new knowledge, exploring new perspectives and tools.
                       As a Web App Developer: Pursue opportunities to build and maintain sophisticated web applications, focusing on creating clean, efficient, and maintainable code.
                       As an Interdisciplinary Data Analyst: Collaborate with data scientists, analysts, and researchers to develop and refine data-driven insights and solutions.
                       As an Entrepreneur: Build and grow businesses that leverage data-driven insights and technologies to achieve success.""",
    'personal_writeup': """Ever Olivares is an accomplished Data Science student with 6 years of experience, specializing in
                        deep learning, and expert in multiple
                        programming languages and frameworks. He holds a BS in applied mathematics, got a fellowship for a PhD and is currently pursuing his master's. He has a strong
                        background in AI and data science. Ever Olivares has successfully led
                        and participated in nationally funded research projects, proving his ability to drive
                        innovation and growth in the tech industry and academia. Ideal for interdisciplinary research
                        roles that require a strategic and innovative approach."""
}

# Kick off the crew
result = job_application_crew.kickoff(inputs=career_counselling_inputs)

print(result)

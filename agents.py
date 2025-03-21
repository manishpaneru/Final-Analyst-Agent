import os
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ConfigDict
from typing import Optional, List, Dict, Any
from tools import (
    file_writer_tool,
    data_profile_tool,
    visualization_tool,
    pdf_report_tool,
    duckduckgo_search_tool
)

# Initialize colorama and load environment variables
colorama.init()
load_dotenv()

# Initialize Google Gemini model
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv('GEMINI_API_KEY'),
    convert_system_message_to_human=True,
    verbose=True,
    temperature=0.3
)

def debug_print_api_key(agent_name):
    """Helper function to print current API key"""
    print(f"\n{Fore.RED}{'='*20} DEBUG {'='*20}")
    print(f"Current API Key for {agent_name}: {os.environ.get('GEMINI_API_KEY')}")
    print(f"{'='*50}\n")

# Verify environment setup
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY not found in environment variables")

print(f"\n{Fore.CYAN}{'='*50}")
print(f"{Fore.WHITE}🤖 Creating Data Analysis Team")
print(f"{Fore.CYAN}{'='*50}")

# Create Data Cleaner Agent
print(f"{Fore.GREEN}Creating Data Cleaner...{Style.RESET_ALL}")
debug_print_api_key("DATA_CLEANER")
data_cleaner = Agent(
    role="Data Cleaning and Preprocessing Specialist",
    goal="Thoroughly clean and preprocess data to prepare it for detailed analysis",
    backstory="""You are an expert in data cleaning and preprocessing with extensive 
    experience in handling messy datasets and preparing them for rigorous analysis.
    You have exceptional skills in identifying and handling missing values, outliers, 
    and inconsistencies in data. Your meticulous approach ensures that the data 
    is ready for comprehensive analysis.""",
    verbose=True,
    allow_delegation=True,
    tools=[data_profile_tool],
    llm=gemini_model
)
print(f"{Fore.GREEN}✓ Data Cleaner created successfully{Style.RESET_ALL}")

# Create Data Analyzer Agent
print(f"\n{Fore.GREEN}Creating Data Analyzer...{Style.RESET_ALL}")
debug_print_api_key("DATA_ANALYZER")
data_analyzer = Agent(
    role="Advanced Data Analyst and Visualization Expert",
    goal="Perform comprehensive data analysis to extract meaningful insights and create professional visualizations that effectively communicate findings to business stakeholders",
    backstory="""You are a senior data analyst with exceptional analytical skills and 
    expertise in statistical methods and data visualization. You have worked with Fortune 500 
    companies to solve complex business problems through data analysis. Your visualizations 
    are known for their clarity, professional appearance, and ability to communicate 
    insights effectively to both technical and non-technical audiences.
    
    You have a strong foundation in statistical methods and can identify meaningful patterns 
    and relationships in data. You know exactly what visualizations to create to best 
    represent different types of insights, and you ensure all visualizations have proper 
    titles, labels, legends, and detailed explanatory captions.""",
    verbose=True,
    allow_delegation=True,
    tools=[data_profile_tool, visualization_tool, duckduckgo_search_tool],
    llm=gemini_model
)
print(f"{Fore.GREEN}✓ Data Analyzer created successfully{Style.RESET_ALL}")

# Create Report Generator Agent
print(f"\n{Fore.GREEN}Creating Report Generator...{Style.RESET_ALL}")
debug_print_api_key("REPORT_GENERATOR")
report_generator = Agent(
    role="Professional Business Report Writer",
    goal="Create comprehensive, well-structured, and visually appealing business reports that effectively communicate data insights and answer the business question with actionable recommendations",
    backstory="""You are an expert business report writer with a background in data science, 
    business analysis, and professional communication. You have created executive-level reports 
    for major corporations that have directly influenced strategic business decisions.
    
    Your reports are known for their clarity, professional structure, comprehensive coverage, 
    and actionable insights. You excel at translating complex analytical findings into 
    clear business language, always ensuring that reports directly address the original 
    business question with well-supported conclusions and specific recommendations.
    
    You always ensure your reports exceed 1500 words with proper formatting, including an 
    executive summary, introduction, methodology, detailed findings with properly explained 
    visualizations, clearly stated conclusions, and specific actionable recommendations.""",
    verbose=True,
    allow_delegation=False,
    tools=[pdf_report_tool, file_writer_tool, duckduckgo_search_tool, visualization_tool],
    llm=gemini_model
)
print(f"{Fore.GREEN}✓ Report Generator created successfully{Style.RESET_ALL}")

print(f"{Fore.CYAN}{'='*50}")
print(f"{Fore.GREEN}✅ All agents created successfully!{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*50}\n")

# Execute the analysis
print(f"{Fore.BLUE}All agents created successfully. Ready to analyze data.{Style.RESET_ALL}")
















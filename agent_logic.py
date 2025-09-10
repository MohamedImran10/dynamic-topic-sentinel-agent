import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

load_dotenv() 

from key_manager import ApiKeyManager
from tools import google_search_tool, scrape_and_analyze_tool, youtube_transcript_tool, pdf_reader_tool
from database import (
    setup_database, check_if_url_exists, add_url, get_cached_report, 
    cache_report, get_seen_urls_for_topic
)

# --- REVISED: PERSONA PROMPTS FOR MORE DIRECT INSTRUCTIONS ---
PERSONA_PROMPTS = {
    "default": (
        "an expert research analyst", 
        "a comprehensive and objective overview of the topic."
    ),
    "marketing_manager": (
        "a senior Marketing Manager", 
        "a concise executive briefing. Focus on market sentiment, competitive landscape, and customer pain points. The tone should be actionable."
    ),
    "academic_researcher": (
        "an Academic Researcher", 
        "a formal literature review. Focus on new findings, methodologies, and potential gaps in current research. The tone must be analytical."
    ),
    "financial_investor": (
        "a Financial Investor", 
        "an investment thesis update. Focus on growth signals, potential risks, and financial implications. The tone must be objective and data-driven."
    ),
    "content_creator": (
        "a Content Creator writing a script", 
        "a set of engaging story notes. Highlight surprising facts, interesting quotes, different angles, and key people involved."
    )
}

# --- This setup code runs once when the application starts ---
setup_database()
key_manager = ApiKeyManager()
all_tools = [google_search_tool, scrape_and_analyze_tool, youtube_transcript_tool, pdf_reader_tool]

def create_gemini_llm(api_key: str, temperature: float = 0.0):
    """Helper function to create a Gemini LLM instance with a specific API key."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_retries=0
    )

def analyze_content_with_retry(url: str, topic: str, analysis_llm_instance):
    # ... (This function is correct and does not need changes)
    content_to_analyze = ""
    if "youtube.com/watch" in url: content_to_analyze = youtube_transcript_tool.func(url)
    elif url.lower().endswith('.pdf'): content_to_analyze = pdf_reader_tool.func(url)
    else: content_to_analyze = scrape_and_analyze_tool.func(url)

    if content_to_analyze and "Could not" not in content_to_analyze and "Failed" not in content_to_analyze:
        add_url(url, topic)
        summary_prompt = f"Summarize the key points of the following content in one paragraph:\n\n---{content_to_analyze}---"
        for _ in range(len(key_manager.keys)):
            try:
                summary = analysis_llm_instance.invoke(summary_prompt).content
                return f"Source: {url}\nSummary: {summary}\n---", analysis_llm_instance
            except ResourceExhausted:
                print(f"❗ Quota exhausted for analysis on key index {key_manager.current_key_index}. Retrying...")
                new_key = key_manager.get_next_key()
                analysis_llm_instance = create_gemini_llm(new_key, temperature=0)
            except Exception as e:
                print(f"-> Analysis failed for {url}: {e}")
                return None, analysis_llm_instance
        print(f"-> Analysis failed for {url}: All API keys are exhausted.")
    return None, analysis_llm_instance

def run_agent_task(topic: str, persona: str = "default") -> dict:
    cached_data = get_cached_report(topic, persona)
    if cached_data:
        return cached_data

    key_manager.current_key_index = 0
    analysis_llm = create_gemini_llm(key_manager.get_current_key(), temperature=0)

    urls_to_process = [url for url in google_search_tool.func(query=topic) if isinstance(url, str)]
    
    all_summaries, successful_urls = [], []

    def process_urls(urls):
        nonlocal analysis_llm
        for url in urls:
            if url not in successful_urls:
                summary_text, updated_llm = analyze_content_with_retry(url, topic, analysis_llm)
                analysis_llm = updated_llm
                if summary_text:
                    all_summaries.append(summary_text)
                    successful_urls.append(url)
    
    process_urls(urls_to_process)

    if not all_summaries:
        seen_urls = get_seen_urls_for_topic(topic)
        if seen_urls:
            process_urls(seen_urls)

    if all_summaries:
        # --- REVISED: A much more direct and robust synthesis prompt ---
        role, task_description = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["default"])
        
        synthesis_prompt = f"""
        Your role is {role}.
        You have been given several research summaries about the topic: '{topic}'.
        Your task is to write {task_description}

        RESEARCH SUMMARIES:
        ---
        {"".join(all_summaries)}
        ---

        Do not mention the summaries or that you are an AI. Write the final report directly.
        """
        
        synthesis_llm = create_gemini_llm(analysis_llm.google_api_key, temperature=0.1)
        
        try:
            # For the final synthesis, we can make a direct call instead of using the agent executor
            # This is faster and more reliable for this specific task.
            final_report_text = synthesis_llm.invoke(synthesis_prompt).content
            
            cache_report(topic, persona, final_report_text, successful_urls)
            return {"report": final_report_text, "sources": successful_urls}
        except Exception as e:
            return {"report": f"Error during final synthesis: {e}", "sources": successful_urls}
    
    return {"report": "No new or recoverable information was found to create a report.", "sources": []}
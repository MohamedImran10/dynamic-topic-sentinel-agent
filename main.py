import os
import time
from dotenv import load_dotenv

load_dotenv() 

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from tools import google_search_tool, scrape_and_analyze_tool, youtube_transcript_tool, pdf_reader_tool
from database import (
    setup_database, 
    check_if_url_exists, 
    add_url, 
    get_cached_report, 
    cache_report,
    get_seen_urls_for_topic
)

def analyze_content(url, topic, analysis_llm):
    """Analyzes content from a URL."""
    # This function is now bug-free and does not check for existing URLs.
    
    print(f"\n🧠 Analyzing URL: {url}")
    
    content_to_analyze = ""
    if "youtube.com/watch" in url:
        content_to_analyze = youtube_transcript_tool.func(url)
    elif url.lower().endswith('.pdf'):
        content_to_analyze = pdf_reader_tool.func(url)
    else:
        content_to_analyze = scrape_and_analyze_tool.func(url)

    if content_to_analyze and "Could not" not in content_to_analyze and "Failed" not in content_to_analyze:
        # We only add the URL to memory after we successfully get content
        add_url(url, topic)
        summary_prompt = f"Summarize the key points of the following content in one paragraph:\n\n---{content_to_analyze}---"
        try:
            summary = analysis_llm.invoke(summary_prompt).content
            print("-> Summary successful.")
            return f"Source: {url}\nSummary: {summary}\n---"
        except Exception as e:
            print(f"-> Analysis failed: {e}")
            return None
    else:
        print(f"-> Skipping analysis due to content extraction failure: {content_to_analyze}")
        return None

def main():
    setup_database()
    
    # Using OpenRouter
    llm = ChatOpenAI(
        model="qwen/qwen3-235b-a22b:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        default_headers={
            "HTTP-Referer": "https://github.com/imran-parthasarathy/dynamic-topic-sentinel",
            "X-Title": "Dynamic Topic Sentinel Agent",
        }
    )
    analysis_llm = ChatOpenAI(
        model="qwen/qwen3-30b-a3b:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/imran-parthasarathy/dynamic-topic-sentinel",
            "X-Title": "Dynamic Topic Sentinel Agent",
        }
    )
    
    all_tools = [google_search_tool, scrape_and_analyze_tool, youtube_transcript_tool, pdf_reader_tool]
    prompt = hub.pull("hwchase17/react")
    synthesis_agent_runnable = create_react_agent(llm, all_tools, prompt)
    synthesis_agent_executor = AgentExecutor(agent=synthesis_agent_runnable, tools=all_tools, verbose=False, handle_parsing_errors=True)

    print("\n--- 🧠 Dynamic Topic Sentinel Agent v8.5 (Typo Fix) ---")
    print("Enter a topic to research, or type 'quit' to exit.")

    while True:
        topic = input("\n> Topic: ")
        if topic.lower() == 'quit':
            print("Exiting agent. Goodbye!"); break

        cached_report = get_cached_report(topic)
        # --- TYPO FIX IS HERE ---
        if cached_report:
            print("\n✅ Found a cached report. Displaying instantly.")
            print("\n--- 📈 Cached Intelligence Briefing ---"); print(cached_report)
            continue

        print(f"\n🔎 No cache found. Searching for new articles on '{topic}'...")
        urls_to_process = [url for url in google_search_tool.func(query=topic) if isinstance(url, str)]
        
        all_summaries = []
        
        # --- Normal Analysis Loop for NEW URLs ---
        for url in urls_to_process:
            if not check_if_url_exists(url, topic): # This is the correct place for the check
                summary_text = analyze_content(url, topic, analysis_llm)
                if summary_text:
                    all_summaries.append(summary_text)
        
        # --- AUTO-RECOVERY LOGIC ---
        if not all_summaries:
            print("\n🤔 No new information found. Checking for a 'stuck' topic to recover...")
            seen_urls = get_seen_urls_for_topic(topic)
            if seen_urls:
                print(f"-> Found {len(seen_urls)} previously seen URLs. Re-analyzing...")
                for url in seen_urls:
                    # The recovery loop will now correctly re-analyze old URLs
                    summary_text = analyze_content(url, topic, analysis_llm)
                    if summary_text:
                        all_summaries.append(summary_text)

        # --- Final Synthesis Step ---
        if all_summaries:
            print("\n\n✨ All sources summarized. Now synthesizing a final report...")
            synthesis_prompt = f"""
            Synthesize the following summaries into a single, cohesive answer for the query: "{topic}"
            Summaries:
            {"\n".join(all_summaries)}
            Provide a final, comprehensive answer. Combine the insights.
            """

            try:
                final_report_obj = synthesis_agent_executor.invoke({"input": synthesis_prompt})
                final_report_text = final_report_obj.get('output', 'Could not generate final report.')
                print("\n--- 📈 Final Intelligence Briefing ---"); print(final_report_text)
                cache_report(topic, final_report_text)
                print("\n✅ Report has been cached for future instant access.")
            except Exception as e:
                print(f"\n--- ❗ Error during final synthesis: {e}")
        else:
            print("\n--- No new or recoverable information was found to create a report. ---")

if __name__ == "__main__":
    main()
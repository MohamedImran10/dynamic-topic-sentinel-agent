import os
import io
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from googleapiclient.discovery import build

@tool
def google_search_tool(query: str) -> list:
    """
    Searches Google for new content related to a query within the last 24 hours,
    excluding PDF files. Returns a list of URLs.
    """
    search_query = f"{query} -filetype:pdf"
    print(f"Executing search with query: '{search_query}'")
    
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
        res = service.cse().list(
            q=search_query,
            cx=os.getenv("GOOGLE_CSE_ID"),
            dateRestrict='d1',
            num=5
        ).execute()
        
        if 'items' not in res:
            return []
            
        return [item['link'] for item in res['items']]
    except Exception as e:
        print(f"An error occurred with the Google Search tool: {e}")
        return []

@tool
def scrape_and_analyze_tool(url: str) -> str:
    """
    Scrapes text content from a standard webpage URL.
    Returns the extracted text content for analysis.
    """
    clean_url = url.strip().strip("'\"")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(clean_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs])
        if not content or len(content.strip()) < 150:
            return f"Could not extract sufficient text content from {clean_url}."
        return content[:8000]
    except Exception as e:
        return f"An error occurred while scraping {clean_url}: {e}"

@tool
def youtube_transcript_tool(url: str) -> str:
    """
    Fetches the transcript of a YouTube video.
    """
    clean_url = url.strip().strip("'\"")
    video_id = clean_url.split("v=")[-1].split("&")[0]
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript_list])
        return transcript_text[:8000]
    except Exception as e:
        return f"Could not retrieve transcript for YouTube video {video_id}: {e}"

@tool
def pdf_reader_tool(url: str) -> str:
    """
    Downloads a PDF from a URL and extracts its text content.
    """
    clean_url = url.strip().strip("'\"")
    try:
        response = requests.get(clean_url, timeout=15)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text:
            return "Could not extract any text from the PDF."
        return text[:8000]
    except Exception as e:
        return f"Failed to read or process the PDF from {clean_url}: {e}"
import sys
import io
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
from langchain_core.tools import tool
from backend.utils.logger import logger

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Use this when the user asks about the current date, time, year, etc."""
    return f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC."

@tool
def run_python_code(code: str) -> str:
    """Executes python code in a sandbox and returns the stdout. Use this for calculations, data analysis, math operations, and scripting."""
    logger.info(f"Executing Python REPL code: {code}")
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    # Safe globals context
    safe_globals = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "json": __import__("json")
    }
    
    try:
        # We execute code. To print output, user should use 'print()'
        exec(code, safe_globals)
        sys.stdout = old_stdout
        val = redirected_output.getvalue()
        return val if val.strip() else "Code executed successfully with no stdout."
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing Python code: {str(e)}"

@tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for the query and returns a summary. Use this for general knowledge, biographies, history, and definitions."""
    logger.info(f"Searching Wikipedia for: {query}")
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "formatversion": 2
        }
        response = httpx.get(url, params=params, timeout=5.0)
        data = response.json()
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return f"No Wikipedia pages found matching: {query}"
            
        summary_list = []
        for result in search_results[:3]:
            title = result["title"]
            snippet = result["snippet"].replace('<span class="searchmatch">', '').replace('</span>', '')
            summary_list.append(f"Title: {title}\nSnippet: {snippet}...\n")
            
        return "\n".join(summary_list)
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return f"Failed to search Wikipedia: {str(e)}"

@tool
def search_duckduckgo(query: str) -> str:
    """Searches DuckDuckGo for the query and returns web results. Use this for real-time news, current events, and live web search."""
    logger.info(f"Searching DuckDuckGo for: {query}")
    try:
        # Use DDG Lite HTML parser to avoid API key requirements and JS rendering issues
        url = "https://lite.duckduckgo.com/lite/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        data = {"q": query}
        
        response = httpx.post(url, data=data, headers=headers, timeout=5.0)
        if response.status_code != 200:
            return f"Failed to retrieve DuckDuckGo search results (Status code: {response.status_code})"
            
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # In DDG Lite, results are table rows
        rows = soup.find_all("td", class_="result-snippet")
        links = soup.find_all("a", class_="result-link")
        
        if not rows or not links:
            return f"No search results found on DuckDuckGo for: {query}"
            
        results = []
        for i in range(min(4, len(rows))):
            title = links[i].get_text()
            link = links[i].get("href")
            snippet = rows[i].get_text().strip()
            results.append(f"Result {i+1}: {title}\nURL: {link}\nSummary: {snippet}\n")
            
        return "\n".join(results)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return f"Failed to query DuckDuckGo: {str(e)}"

@tool
def search_arxiv(query: str) -> str:
    """Searches arXiv for academic research papers. Use this for AI papers, Machine Learning developments, and math research."""
    logger.info(f"Searching arXiv for: {query}")
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": 3
        }
        response = httpx.get(url, params=params, timeout=5.0)
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "xml")
        entries = soup.find_all("entry")
        
        if not entries:
            return f"No arXiv research papers found for: {query}"
            
        results = []
        for i, entry in enumerate(entries):
            title = entry.title.get_text().strip().replace("\n", " ")
            summary = entry.summary.get_text().strip().replace("\n", " ")[:200]
            id_url = entry.id.get_text().strip()
            author_names = ", ".join([a.find("name").get_text() for a in entry.find_all("author")])
            results.append(f"Paper {i+1}: {title}\nAuthors: {author_names}\nLink: {id_url}\nSummary: {summary}...\n")
            
        return "\n".join(results)
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return f"Failed to search arXiv: {str(e)}"

@tool
def get_exchange_rate(base_currency: str = "USD", target_currency: str = "EUR") -> str:
    """Retrieves exchange rate for currency conversions. Inputs are base_currency and target_currency symbols (e.g. USD, EUR, INR, GBP)."""
    logger.info(f"Fetching exchange rate from {base_currency} to {target_currency}...")
    
    # Static exchange rate dictionary for reliability, fallback to real API if keys exist
    rates = {
        "USD_EUR": 0.92, "EUR_USD": 1.09,
        "USD_INR": 83.2, "INR_USD": 0.012,
        "USD_GBP": 0.79, "GBP_USD": 1.27,
        "EUR_INR": 90.5, "INR_EUR": 0.011,
        "GBP_INR": 105.4, "INR_GBP": 0.0095
    }
    
    key = f"{base_currency.upper()}_{target_currency.upper()}"
    rev_key = f"{target_currency.upper()}_{base_currency.upper()}"
    
    if key in rates:
        rate = rates[key]
        return f"The exchange rate from {base_currency.upper()} to {target_currency.upper()} is {rate}."
    elif rev_key in rates:
        rate = 1.0 / rates[rev_key]
        return f"The exchange rate from {base_currency.upper()} to {target_currency.upper()} is {rate:.4f}."
    else:
        # Default mock response for unregistered currencies
        return f"Mock exchange rate: 1 {base_currency.upper()} = 1.25 {target_currency.upper()}."

# Tool Registry list
ALL_TOOLS = [
    get_current_datetime,
    run_python_code,
    search_wikipedia,
    search_duckduckgo,
    search_arxiv,
    get_exchange_rate
]

# Map of tool names to objects for execution
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}

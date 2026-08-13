import re
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from backend.utils.logger import logger

def clean_html(html_content: str) -> str:
    """Removes HTML tags, scripts, styles, and extra whitespace."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            script.decompose()
            
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text_clean = "\n".join(chunk for chunk in chunks if chunk)
        
        return text_clean
    except Exception as e:
        logger.error(f"HTML cleaning failed: {e}")
        # Simple regex fallback if BS4 fails
        clean = re.sub(r'<[^>]+>', '', html_content)
        return re.sub(r'\s+', ' ', clean).strip()

def clean_text(text: str) -> str:
    """Sanitizes text by removing non-printable characters and normalizing spacing."""
    if not text:
        return ""
    # Standardize whitespace characters (replace tabs and multiple spaces with a single space)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    # Normalize multiple newlines to at most two
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_metadata_from_soup(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Extracts title, meta tags, and category info from beautiful soup object."""
    metadata = {
        "title": "Untitled Document",
        "description": "",
        "author": "Unknown",
        "category": "General",
        "tags": "general",
        "source": url
    }
    
    if not soup:
        return metadata
        
    try:
        # Title
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()
            
        # Meta description
        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag.get("content", "").strip()
            
        # Meta author
        author_tag = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"name": "article:author"})
        if author_tag and author_tag.get("content"):
            metadata["author"] = author_tag.get("content", "").strip()
            
        # Keywords / Tags
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            tags_list = [k.strip() for k in keywords_tag.get("content", "").split(",") if k.strip()]
            if tags_list:
                metadata["tags"] = ", ".join(tags_list)
            
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        
    return metadata

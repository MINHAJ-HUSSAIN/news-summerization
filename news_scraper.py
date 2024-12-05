import requests
from newspaper import Article
from langchain.schema import Document
import time

def newsscrapper(url, url_number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()
    try:
        response = session.get(url, headers=headers, timeout=10)
        status = response.status_code
        if status == 200:
            article = Article('')
            article.set_html(response.text)
            article.parse()
            news_text = article.text.strip()
            news_authors = ','.join(article.authors)
            news_title = article.title
            news_publication_date = str(article.publish_date)

            data = f"Article Title: {news_title}\nPublication Date: {news_publication_date}\n{news_text}"
            metadata = {
                "source": f"{url}",
                "Author": f"{news_authors}",
                "Publication date": f"{news_publication_date}",
                "Article Title": f"{news_title}",
            }
            return Document(page_content=data, metadata=metadata)
        else:
            return None
    except Exception as e:
        print(f"Error scraping URL {url_number}: {e}")
        return None

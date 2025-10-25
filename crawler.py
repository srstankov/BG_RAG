import re
import json
import scrapy
import requests
from bs4 import BeautifulSoup
import html
from requests.adapters import HTTPAdapter, Retry

requests_session = requests.Session()

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])

requests_session.mount('https://', HTTPAdapter(max_retries=retries))

URL_TMPL = "https://www.focus-news.net/novini/{}.html"

LAST_ARTICLE = 2758000
FIRST_ARTICLE = 2700000
DELAY = 0.5

urls = [URL_TMPL.format(x)
        for x in range(int(FIRST_ARTICLE), int(LAST_ARTICLE+1))]

real_urls = []

for url in urls:
    r = requests_session.get(url, allow_redirects=True, timeout=10)
    if r.status_code == 200 and not r.url == "https://www.focus-news.net/" and \
            not r.url == "https://www.focus-news.net/sport":
        real_urls.append(url)


class FocusNewsSpider(scrapy.Spider):
    name = "focusnews"
    download_delay = DELAY
    custom_settings = {
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def __init__(self, first_article=FIRST_ARTICLE, last_article=LAST_ARTICLE):
            self.start_urls = real_urls

    def parse(self, response):
        url = str(response.xpath("//meta[@property='og:url']/@content").extract_first())

        if url.startswith('https://www.focus-news.net/sport/'):
            json_ld = response.xpath('//script[@type="application/ld+json"]/text()').extract_first()
            id = url[-12:-5]
        else:
            json_ld = response.xpath('//script[@type="application/ld+json"]/text()').extract()[1]
            id = url[-7:]

        json_response = json.loads(json_ld)

        title = str(json_response['headline'])
        title = title.replace("“", '"').replace('&quot;', '"').replace(u'\xa0', u' ')
        title = BeautifulSoup(html.unescape(title)).get_text()

        text = str(json_response['articleBody'])
        text = text.replace("“", '"').replace('&quot;', '"').replace(u'\xa0', u' ')
        text = re.sub(r'([.?!:;/\""])([А-ЯA-Z])', r'\1 \2', text)
        text = re.sub(r'([а-яa-z])([А-ЯA-Z])', r'\1 \2', text)
        text = re.sub(r'([1-9])([А-ЯA-Z])', r'\1 \2', text)
        text = ' '.join(text.split())
        text = BeautifulSoup(html.unescape(text)).get_text()

        keywords = str(json_response['keywords'])
        keywords = keywords.replace("“", '"').replace('&quot;', '"')

        yield {
            'id': id,
            'url': url,
            'title': title,
            'text': text,
            'keywords': keywords
        }

# example command to run
# scrapy runspider crawler.py -o focus_news_articles_0_0.json

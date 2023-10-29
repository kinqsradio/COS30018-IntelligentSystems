import feedparser
from datetime import datetime, timedelta

class YahooFinanceNews:

    def __init__(self):
        self.rss = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'
        
    def fetch_news(self, ticker, days=7):

        # Parse the RSS feed
        feed = feedparser.parse(self.rss % ticker)

        # Get the current date
        current_date = datetime.now()

        # Filter entries from the past 'days' days
        recent_entries = [entry for entry in feed.entries if (current_date - datetime(*entry.published_parsed[:6])) <= timedelta(days=days)]

        return recent_entries
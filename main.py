import os
import argparse
import json
import logging
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configuration
LOG_FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class TextAnalyzer:
    def __init__(self):
        self._setup_resources()
        self.stop_words = set(stopwords.words("english")) | set(stopwords.words("turkish"))
        self.sia = SentimentIntensityAnalyzer()

    def _setup_resources(self):
        """Ensure all NLP resources are available."""
        resources = ['punkt', 'stopwords', 'vader_lexicon', 'punkt_tab']
        for res in resources:
            try:
                nltk.data.find(res)
            except LookupError:
                logging.info(f"Downloading NLTK resource: {res}")
                nltk.download(res, quiet=True)

    def process_text(self, text):
        """Cleans and tokenizes text."""
        tokens = word_tokenize(text.lower())
        words = [w for w in tokens if w.isalpha()]
        filtered = [w for w in words if w not in self.stop_words]
        return words, filtered

    def analyze(self, text, top_n=10):
        """Performs full NLP analysis."""
        words, filtered = self.process_text(text)
        
        # Frequency Analysis
        counter = Counter(filtered)
        top_words = counter.most_common(top_n)

        # Sentiment Analysis
        sentiment = self.sia.polarity_scores(text)

        return {
            "metrics": {
                "total_words": len(words),
                "unique_words": len(set(words)),
                "sentiment": sentiment
            },
            "top_words": top_words,
            "cleaned_text": " ".join(filtered)
        }

    def generate_visuals(self, results, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Bar Chart
        words, counts = zip(*results["top_words"])
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts, color='teal')
        plt.title("Top Word Frequencies")
        plt.savefig(f"{output_dir}/frequency_chart.png")
        
        # 2. Word Cloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(results["cleaned_text"])
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f"{output_dir}/wordcloud.png")
        
        logging.info(f"Visuals saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="AI Text Analytics Suite")
    parser.add_argument("--file", type=str, help="Path to input .txt file")
    parser.add_argument("--visualize", action="store_true", help="Generate charts and word clouds")
    
    args = parser.parse_args()
    analyzer = TextAnalyzer()

    try:
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = input("Paste text to analyze: ")

        results = analyzer.analyze(content)

        # Print Clean Results
        print("\n" + "="*30)
        print("   TEXT ANALYSIS REPORT")
        print("="*30)
        print(f"Total Words:     {results['metrics']['total_words']}")
        print(f"Sentiment Score: {results['metrics']['sentiment']['compound']} (-1 to 1)")
        print("-" * 30)
        for word, count in results['top_words']:
            print(f"{word.ljust(15)} : {count}")

        if args.visualize:
            analyzer.generate_visuals(results)

    except Exception as e:
        logging.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()

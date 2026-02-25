"""
AI-Powered Text Analytics Suite
================================
A professional-grade NLP tool for deep text analysis including
sentiment scoring, frequency analysis, and visual intelligence.

Author  : Merve Sude Böler
GitHub  : https://github.com/mervesudeboler
License : MIT
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from collections import Counter
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ─────────────────────────────────────────
#  Logging Configuration
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  Core Analyzer
# ─────────────────────────────────────────
class TextAnalyzer:
    """
    Multi-lingual NLP analysis engine.

    Capabilities
    ------------
    - Tokenisation & stop-word removal (English + Turkish)
    - VADER-based sentiment scoring
    - Word-frequency ranking
    - Bar-chart & word-cloud generation
    - JSON report export
    """

    NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "vader_lexicon"]

    def __init__(self) -> None:
        self._bootstrap_nltk()
        self.stop_words = (
            set(stopwords.words("english")) | set(stopwords.words("turkish"))
        )
        self.sia = SentimentIntensityAnalyzer()
        logger.info("TextAnalyzer initialised.")

    # Private helpers

    def _bootstrap_nltk(self) -> None:
        """Download any missing NLTK resources silently."""
        for resource in self.NLTK_RESOURCES:
            try:
                nltk.data.find(resource)
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)

    @staticmethod
    def _sentiment_label(compound: float) -> str:
        """Map a VADER compound score to a human-readable label."""
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    # Public API

    def process_text(self, text: str) -> tuple:
        """
        Lowercase, tokenise, and filter a raw text string.

        Returns
        -------
        words     : all alphabetic tokens (before stop-word removal)
        filtered  : tokens after stop-word removal
        """
        tokens = word_tokenize(text.lower())
        words = [w for w in tokens if w.isalpha()]
        filtered = [w for w in words if w not in self.stop_words]
        return words, filtered

    def analyze(self, text: str, top_n: int = 15) -> dict:
        """
        Run a full NLP analysis pipeline on text.

        Parameters
        ----------
        text  : raw input string
        top_n : number of top words to surface

        Returns
        -------
        Structured result dict with metrics, top_words, and cleaned_text.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        words, filtered = self.process_text(text)
        counter = Counter(filtered)
        top_words = counter.most_common(top_n)
        sentiment = self.sia.polarity_scores(text)

        return {
            "metadata": {
                "analysed_at": datetime.utcnow().isoformat() + "Z",
                "input_characters": len(text),
            },
            "metrics": {
                "total_words": len(words),
                "unique_words": len(set(words)),
                "filtered_words": len(filtered),
                "lexical_density": (
                    round(len(set(words)) / len(words), 4) if words else 0
                ),
                "sentiment": {
                    **sentiment,
                    "label": self._sentiment_label(sentiment["compound"]),
                },
            },
            "top_words": top_words,
            "cleaned_text": " ".join(filtered),
        }

    def generate_visuals(
        self, results: dict, output_dir: str = "outputs"
    ) -> None:
        """Render and save a frequency bar-chart and a word-cloud PNG."""
        os.makedirs(output_dir, exist_ok=True)

        if not results["top_words"]:
            logger.warning("No words available - skipping visuals.")
            return

        words, counts = zip(*results["top_words"])

        # Bar Chart
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(words, counts, color="#4A90D9", edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, padding=3, fontsize=9, color="#333333")
        ax.set_title("Top Word Frequencies", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Word", fontsize=11)
        ax.set_ylabel("Occurrences", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "frequency_chart.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Bar chart saved -> %s", chart_path)

        # Word Cloud
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="Blues",
            max_words=100,
            collocations=False,
        ).generate(results["cleaned_text"])

        fig2, ax2 = plt.subplots(figsize=(15, 7))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        plt.tight_layout(pad=0)
        wc_path = os.path.join(output_dir, "wordcloud.png")
        fig2.savefig(wc_path, dpi=150)
        plt.close(fig2)
        logger.info("Word cloud saved -> %s", wc_path)

    def export_json(self, results: dict, output_dir: str = "outputs") -> str:
        """Serialise the analysis results to a timestamped JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{ts}.json")

        export_data = {
            **results,
            "top_words": [
                {"word": w, "count": c} for w, c in results["top_words"]
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(export_data, fh, indent=2, ensure_ascii=False)

        logger.info("JSON report saved -> %s", path)
        return path


# ─────────────────────────────────────────
#  CLI Presentation Helpers
# ─────────────────────────────────────────

def _print_report(results: dict) -> None:
    """Pretty-print the analysis report to stdout."""
    m = results["metrics"]
    sep = "-" * 40

    print("\n" + "=" * 40)
    print("   TEXT ANALYSIS REPORT")
    print("=" * 40)
    print(f"  Analysed at    : {results['metadata']['analysed_at']}")
    print(sep)
    print(f"  Total words    : {m['total_words']}")
    print(f"  Unique words   : {m['unique_words']}")
    print(f"  Lexical density: {m['lexical_density']:.1%}")
    print(sep)
    s = m["sentiment"]
    print(f"  Sentiment      : {s['label']}")
    print(f"  Compound score : {s['compound']:+.4f}  (range -1 to +1)")
    print(f"  Positive       : {s['pos']:.1%}")
    print(f"  Neutral        : {s['neu']:.1%}")
    print(f"  Negative       : {s['neg']:.1%}")
    print(sep)
    print("  Top Words:")
    for rank, (word, count) in enumerate(results["top_words"], start=1):
        bar = "#" * min(count, 30)
        print(f"   {rank:>2}. {word:<18} {count:>4}  {bar}")
    print("=" * 40 + "\n")


# ─────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-text-analyzer",
        description="AI-Powered Text Analytics Suite - NLP analysis in seconds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --file article.txt --visualize\n"
            "  python main.py --file article.txt --export-json\n"
            "  echo 'I love NLP' | python main.py\n"
        ),
    )
    parser.add_argument("--file", type=str, metavar="PATH",
                        help="Path to a plain-text (.txt) input file")
    parser.add_argument("--top-n", type=int, default=15, metavar="N",
                        help="Number of top words to display (default: 15)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate bar chart and word cloud images")
    parser.add_argument("--export-json", action="store_true",
                        help="Export the full report as a JSON file")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        metavar="DIR", help="Directory for all output files (default: outputs/)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    analyzer = TextAnalyzer()

    try:
        if args.file:
            if not os.path.isfile(args.file):
                logger.error("File not found: %s", args.file)
                sys.exit(1)
            with open(args.file, "r", encoding="utf-8") as fh:
                content = fh.read()
            logger.info("Loaded file: %s (%d chars)", args.file, len(content))
        elif not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            print("Paste your text below (press Ctrl+D / Ctrl+Z when done):")
            content = sys.stdin.read()

        results = analyzer.analyze(content, top_n=args.top_n)
        _print_report(results)

        if args.visualize:
            analyzer.generate_visuals(results, output_dir=args.output_dir)

        if args.export_json:
            analyzer.export_json(results, output_dir=args.output_dir)

    except ValueError as exc:
        logger.error("Input error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()

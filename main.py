import os
import argparse
import json
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def analyze_text(text, top_n=10, plot=False, export=None):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]

    stop_words = set(stopwords.words("turkish")) | set(stopwords.words("english"))
    filtered = [w for w in words if w not in stop_words]

    counter = Counter(filtered)

    result = {
        "word_count": len(words),
        "unique_words": len(set(words)),
        "top_words": counter.most_common(top_n)
    }

    if export in ("json", "both"):
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if export in ("csv", "both"):
        with open("outputs/keywords.csv", "w", encoding="utf-8") as f:
            f.write("word,count\n")
            for w, c in result["top_words"]:
                f.write(f"{w},{c}\n")

    if plot:
        words_, counts_ = zip(*result["top_words"])
        plt.figure(figsize=(10,5))
        plt.bar(words_, counts_)
        plt.xticks(rotation=45)
        plt.title("Top Words")
        plt.tight_layout()
        plt.savefig("outputs/keywords.png")

    return result


def main():
    parser = argparse.ArgumentParser(description="Advanced AI Text Analyzer (NLP)")
    parser.add_argument("--file", type=str, help="Input text file")
    parser.add_argument("--top", type=int, default=10, help="Top N words")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    parser.add_argument("--export", choices=["json", "csv", "both"], help="Export results")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = input("Enter text: ")

    result = analyze_text(
        text,
        top_n=args.top,
        plot=args.plot,
        export=args.export
    )

    print("Word count:", result["word_count"])
    print("Unique words:", result["unique_words"])


if __name__ == "__main__":
    main()

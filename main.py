def analyze_text(text):
    words = text.lower().split()
    return {
        "word_count": len(words),
        "unique_words": len(set(words))
    }

if __name__ == "__main__":
    text = input("Enter text: ")
    result = analyze_text(text)

    print("Word count:", result["word_count"])
    print("Unique words:", result["unique_words"])

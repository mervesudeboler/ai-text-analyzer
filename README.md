# 🤖 AI-Powered Text Analytics Suite

> A professional-grade, multi-lingual NLP toolkit for deep text analysis — built with Python & NLTK.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Sentiment Analysis** | VADER lexicon scores — Positive / Neutral / Negative with compound rating |
| 🌍 **Multilingual Filtering** | Stop-word removal for both **English** and **Turkish** |
| 📊 **Word Frequency Ranking** | Top-N most frequent words with inline bar visualisation |
| 📈 **Lexical Density** | Measures vocabulary richness (unique ÷ total words) |
| 🖼️ **Visual Intelligence** | Exports high-res **bar charts** and **word clouds** (PNG, 150 DPI) |
| 💾 **JSON Export** | Timestamped, machine-readable reports for downstream pipelines |
| 🧱 **OOP Architecture** | Modular `TextAnalyzer` class — easy to extend and integrate |

---

## 🛠️ Installation

> **Requirements:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/mervesudeboler/ai-text-analyzer.git
cd ai-text-analyzer

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Analyse a text file
```bash
python main.py --file article.txt
```

### Analyse a file + generate visuals
```bash
python main.py --file article.txt --visualize
```

### Full pipeline — visuals + JSON export
```bash
python main.py --file article.txt --visualize --export-json
```

### Pipe text directly from the terminal
```bash
echo "Natural Language Processing is a fascinating field." | python main.py
```

### Interactive mode (no arguments)
```bash
python main.py
# Paste your text, then press Ctrl+D
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--file PATH` | — | Path to a `.txt` input file |
| `--top-n N` | `15` | Number of top words to display |
| `--visualize` | off | Generate bar chart & word cloud |
| `--export-json` | off | Save full report as JSON |
| `--output-dir DIR` | `outputs/` | Destination for all output files |

---

## 📦 Output Structure

```
outputs/
├── frequency_chart.png          # Bar chart of top-N word frequencies
├── wordcloud.png                # Visual word cloud
└── report_YYYYMMDD_HHmmSS.json  # Full analysis report (with --export-json)
```

### Sample JSON Report
```json
{
  "metadata": {
    "analysed_at": "2025-02-25T10:00:00Z",
    "input_characters": 1024
  },
  "metrics": {
    "total_words": 180,
    "unique_words": 95,
    "lexical_density": 0.5278,
    "sentiment": {
      "compound": 0.7269,
      "label": "Positive",
      "pos": 0.214,
      "neu": 0.786,
      "neg": 0.0
    }
  },
  "top_words": [
    { "word": "language", "count": 12 },
    { "word": "model", "count": 9 }
  ]
}
```

---

## 🏗️ Project Structure

```
ai-text-analyzer/
├── main.py            # Core NLP engine + CLI
├── requirements.txt   # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📚 Tech Stack

- **[NLTK](https://www.nltk.org/)** — Tokenisation, stop-word removal, VADER sentiment
- **[Matplotlib](https://matplotlib.org/)** — Bar chart visualisation
- **[WordCloud](https://github.com/amueller/word_cloud)** — Word cloud generation

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by <a href="https://github.com/mervesudeboler">Merve Sude Böler</a></p>

# Twitter Sentiment Analysis Tool 🐦

A simple sentiment analysis tool that classifies tweets as positive, negative, or neutral using NLP.

## Project Structure

```
Twitter Sentiment Analysis/
├── sentiment_analyzer.py   # Main analysis script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

- **Two Analysis Methods**: Uses both VADER and TextBlob for comparison
- **Tweet Preprocessing**: Cleans URLs, mentions, hashtags, and special characters
- **Interactive Mode**: Analyze your own tweets in real-time
- **Sample Dataset**: Includes 15 sample tweets for demonstration
- **Visualization**: Generates pie charts showing sentiment distribution
- **Detailed Scores**: Shows confidence scores for each analysis

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Analyzer

```bash
python sentiment_analyzer.py
```

### Step 3: Choose an Option

1. **Analyze sample tweets** - Uses built-in sample data
2. **Enter your own tweets** - Interactive mode
3. **Analyze a single tweet** - Quick single analysis

## How It Works

### 1. Text Preprocessing
- Removes URLs, @mentions, and special characters
- Cleans hashtag symbols (keeps the word)
- Removes retweet indicators (RT)

### 2. VADER Sentiment Analysis
- Uses NLTK's VADER (Valence Aware Dictionary for Sentiment Reasoning)
- Specifically designed for social media text
- Returns compound score (-1 to +1)
- Classification:
  - **Positive**: compound >= 0.05
  - **Negative**: compound <= -0.05
  - **Neutral**: -0.05 < compound < 0.05

### 3. TextBlob Analysis
- Uses TextBlob's pattern analyzer
- Returns polarity score (-1 to +1)
- Classification:
  - **Positive**: polarity > 0.1
  - **Negative**: polarity < -0.1
  - **Neutral**: -0.1 <= polarity <= 0.1

## Example Output

```
==================================================
    TWITTER SENTIMENT ANALYSIS TOOL
    Using NLTK VADER & TextBlob
==================================================

TWEET ANALYSIS
==================================================
Original: I love this product! It's absolutely amazing! 😍
Cleaned:  I love this product Its absolutely amazing

--- VADER Analysis ---
Sentiment: Positive
Scores: Positive=0.567, Negative=0.000, Neutral=0.433
Compound Score: 0.879

--- TextBlob Analysis ---
Sentiment: Positive
Polarity: 0.625
```

## Sample Results

| Tweet | VADER | TextBlob |
|-------|-------|----------|
| "I love this product!" | Positive | Positive |
| "This is terrible..." | Negative | Negative |
| "Meeting at 3 PM" | Neutral | Neutral |

## Technologies Used

- **Python 3.8+**
- **NLTK** - Natural Language Toolkit with VADER
- **TextBlob** - Simplified text processing
- **pandas** - Data manipulation
- **matplotlib** - Data visualization

## NLP Concepts Used

1. **Tokenization** - Breaking text into words
2. **Sentiment Lexicons** - Pre-built word sentiment scores
3. **Polarity Detection** - Measuring positive/negative sentiment
4. **Text Normalization** - Cleaning and standardizing text

## Output Files

- `sentiment_distribution.png` - Pie charts showing sentiment breakdown

## Notes

⚠️ **Limitations**:
- Sentiment analysis may not capture sarcasm or context
- Works best with English text
- Social media slang may affect accuracy

## Author

Intern Project - Twitter Sentiment Analysis

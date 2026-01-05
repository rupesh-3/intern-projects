"""
Twitter Sentiment Analysis Tool
Classifies tweets as positive, negative, or neutral using NLP.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import re
import string


def download_nltk_data():
    """Download required NLTK data files if not already present."""
    resources = ['vader_lexicon', 'punkt', 'stopwords']
    missing = []
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                          f'sentiment/{resource}' if resource == 'vader_lexicon' else
                          f'corpora/{resource}')
        except LookupError:
            missing.append(resource)
    
    if missing:
        print("Downloading NLTK data...")
        for resource in missing:
            nltk.download(resource, quiet=True)
        print("NLTK data downloaded successfully!")
    else:
        print("NLTK data already available.")


# Create a single instance of SentimentIntensityAnalyzer for efficiency
_sia = None

def get_sentiment_analyzer():
    """Get or create the SentimentIntensityAnalyzer singleton."""
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


def clean_tweet(tweet):
    """
    Clean and preprocess tweet text.
    
    Args:
        tweet: Raw tweet text
    
    Returns:
        Cleaned tweet text
    """
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove @mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtag symbol (keep the word)
    tweet = re.sub(r'#', '', tweet)
    
    # Remove RT (retweet indicator)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    
    # Remove special characters and numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    
    return tweet.strip()


def analyze_sentiment_vader(tweet):
    """
    Analyze sentiment using NLTK's VADER.
    
    Args:
        tweet: Tweet text
    
    Returns:
        Sentiment label and scores
    """
    sia = get_sentiment_analyzer()
    scores = sia.polarity_scores(tweet)
    
    # Classify based on compound score
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, scores


def analyze_sentiment_textblob(tweet):
    """
    Analyze sentiment using TextBlob.
    
    Args:
        tweet: Tweet text
    
    Returns:
        Sentiment label and polarity score
    """
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, polarity


def analyze_single_tweet(tweet):
    """
    Analyze a single tweet with both methods.
    
    Args:
        tweet: Tweet text
    """
    cleaned = clean_tweet(tweet)
    
    print("\n" + "="*60)
    print("TWEET ANALYSIS")
    print("="*60)
    print(f"\nOriginal: {tweet}")
    print(f"Cleaned:  {cleaned}")
    
    # VADER Analysis
    vader_sentiment, vader_scores = analyze_sentiment_vader(cleaned)
    print(f"\n--- VADER Analysis ---")
    print(f"Sentiment: {vader_sentiment}")
    print(f"Scores: Positive={vader_scores['pos']:.3f}, "
          f"Negative={vader_scores['neg']:.3f}, "
          f"Neutral={vader_scores['neu']:.3f}")
    print(f"Compound Score: {vader_scores['compound']:.3f}")
    
    # TextBlob Analysis
    blob_sentiment, blob_polarity = analyze_sentiment_textblob(cleaned)
    print(f"\n--- TextBlob Analysis ---")
    print(f"Sentiment: {blob_sentiment}")
    print(f"Polarity: {blob_polarity:.3f}")
    
    return vader_sentiment, blob_sentiment


def analyze_multiple_tweets(tweets):
    """
    Analyze multiple tweets and show statistics.
    
    Args:
        tweets: List of tweet texts
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for tweet in tweets:
        cleaned = clean_tweet(tweet)
        vader_sentiment, vader_scores = analyze_sentiment_vader(cleaned)
        blob_sentiment, blob_polarity = analyze_sentiment_textblob(cleaned)
        
        results.append({
            'Tweet': tweet[:50] + '...' if len(tweet) > 50 else tweet,
            'Cleaned': cleaned[:50] + '...' if len(cleaned) > 50 else cleaned,
            'VADER': vader_sentiment,
            'VADER_Score': vader_scores['compound'],
            'TextBlob': blob_sentiment,
            'TextBlob_Score': blob_polarity
        })
    
    df = pd.DataFrame(results)
    return df


def plot_sentiment_distribution(df):
    """
    Plot sentiment distribution.
    
    Args:
        df: DataFrame with sentiment results
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # VADER Distribution
    vader_counts = df['VADER'].value_counts()
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    vader_colors = [colors.get(x, '#95a5a6') for x in vader_counts.index]
    
    axes[0].pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%',
                colors=vader_colors, startangle=90)
    axes[0].set_title('VADER Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # TextBlob Distribution
    blob_counts = df['TextBlob'].value_counts()
    blob_colors = [colors.get(x, '#95a5a6') for x in blob_counts.index]
    
    axes[1].pie(blob_counts, labels=blob_counts.index, autopct='%1.1f%%',
                colors=blob_colors, startangle=90)
    axes[1].set_title('TextBlob Sentiment Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'sentiment_distribution.png'")


def print_summary(df):
    """
    Print summary statistics.
    
    Args:
        df: DataFrame with sentiment results
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Tweets Analyzed: {len(df)}")
    
    print("\n--- VADER Results ---")
    vader_counts = df['VADER'].value_counts()
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = vader_counts.get(sentiment, 0)
        pct = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print("\n--- TextBlob Results ---")
    blob_counts = df['TextBlob'].value_counts()
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = blob_counts.get(sentiment, 0)
        pct = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print(f"\nAverage VADER Score: {df['VADER_Score'].mean():.3f}")
    print(f"Average TextBlob Score: {df['TextBlob_Score'].mean():.3f}")


# Sample tweets for demonstration
SAMPLE_TWEETS = [
    "I love this product! It's absolutely amazing and exceeded my expectations! 😍",
    "This is the worst experience ever. Totally disappointed and frustrated. 😡",
    "Just had lunch. It was okay, nothing special.",
    "Can't believe how great this new update is! Finally they listened to users!",
    "The customer service was terrible. Never buying from them again.",
    "Weather is nice today. Going for a walk.",
    "So excited for the weekend! Best feeling ever! 🎉",
    "This movie was boring and a complete waste of time.",
    "The meeting has been rescheduled to 3 PM tomorrow.",
    "Thank you so much for your help! You're the best! ❤️",
    "I'm really angry about this situation. Unacceptable!",
    "Just another regular Monday at work.",
    "Wow! This is incredible news! So happy right now!",
    "Feeling sad and lonely today. Not a good day.",
    "The new iPhone looks interesting. Might consider buying it.",
]


def interactive_mode():
    """Run interactive mode for analyzing custom tweets."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter tweets to analyze (type 'quit' to exit)")
    print("Or press Enter with no input to analyze sample tweets")
    
    while True:
        print("\n" + "-"*40)
        tweet = input("Enter tweet: ").strip()
        
        if tweet.lower() == 'quit':
            print("Exiting interactive mode...")
            break
        
        if not tweet:
            print("\nNo input. Analyzing sample tweets instead...")
            return False
        
        analyze_single_tweet(tweet)
    
    return True


def main():
    """Main function to run the sentiment analyzer."""
    print("="*60)
    print("    TWITTER SENTIMENT ANALYSIS TOOL")
    print("    Using NLTK VADER & TextBlob")
    print("="*60)
    
    # Download required NLTK data
    download_nltk_data()
    
    # Ask user for mode
    print("\nChoose an option:")
    print("1. Analyze sample tweets")
    print("2. Enter your own tweets")
    print("3. Analyze a single tweet")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '3':
        tweet = input("\nEnter your tweet: ").strip()
        if tweet:
            analyze_single_tweet(tweet)
        else:
            print("No tweet entered. Using a sample tweet.")
            analyze_single_tweet(SAMPLE_TWEETS[0])
    
    elif choice == '2':
        if interactive_mode():
            return
        # If user pressed Enter without input, fall through to sample analysis
        choice = '1'
    
    if choice == '1' or choice == '':
        print("\nAnalyzing sample tweets...")
        
        # Analyze all sample tweets
        df = analyze_multiple_tweets(SAMPLE_TWEETS)
        
        # Display results table
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        pd.set_option('display.max_colwidth', 50)
        print(df[['Tweet', 'VADER', 'TextBlob']].to_string(index=True))
        
        # Print summary
        print_summary(df)
        
        # Plot distribution
        try:
            plot_sentiment_distribution(df)
        except Exception as e:
            print(f"\nCould not display plot: {e}")
            print("Results saved to 'sentiment_distribution.png'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

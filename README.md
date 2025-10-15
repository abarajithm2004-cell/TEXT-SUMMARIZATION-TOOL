# text_summarizer.py

# Required libraries
import nltk
import heapq
import re

# Download required nltk packages (run once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def summarize_text(text, summary_length=3):
    # Text preprocessing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    
    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))

    # Calculate word frequencies
    word_freq = {}
    for word in words:
        if word.isalnum() and word not in stop_words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    # Normalize word frequencies
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    # Score sentences based on word frequency
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                if len(sent.split(' ')) < 30:  # Ignore very long sentences
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word]
                    else:
                        sentence_scores[sent] += word_freq[word]

    # Get top sentences for summary
    summary_sentences = heapq.nlargest(summary_length, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


# --------- Example Usage ---------

if __name__ == "__main__":
    print("\n=== Text Summarization Tool ===\n")

    # Input: Sample long text
    input_text = """
    Artificial Intelligence (AI) is rapidly transforming various industries, from healthcare and finance to education and transportation.
    The integration of AI technologies, such as machine learning and natural language processing, enables businesses to automate processes,
    enhance decision-making, and provide personalized experiences to users. However, this growth also raises concerns about job displacement,
    data privacy, and ethical use of AI systems. Governments and organizations worldwide are working to establish regulations and frameworks
    to ensure responsible development and deployment of AI. As the technology continues to evolve, it holds the potential to solve some of
    humanity’s most pressing challenges, including climate change, disease diagnosis, and disaster response.
    """

    # Generate summary
    summary = summarize_text(input_text, summary_length=2)

    # Output
    print("\n--- Original Text ---\n")
    print(input_text)

    print("\n--- Summary ---\n")
    print(summary)

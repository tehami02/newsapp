from flask import Flask, render_template, request, redirect, url_for, Response,session
import requests
import csv
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import feedparser
import base64
from io import BytesIO,StringIO
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from wordcloud import WordCloud
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation


app = Flask(__name__)
app.secret_key = '89765thio8'
# Download NLTK data
nltk.download('wordnet')

def lemmatize_stemming(text):
    return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    source = request.form['source']
    return redirect(url_for('display_news', source=source))

def get_news_articles(source_url, num_articles=100):
    # Parse the RSS feed
    feed = feedparser.parse(source_url)
    articles = []
    
    # Iterate over the feed entries and collect articles
    for entry in feed.entries[:num_articles]:
        article = {
            'title': entry.title,
            'link': entry.link,
            'description': entry.description
        }
        articles.append(article)
    
    return articles

@app.route('/display_news/<source>')
def display_news(source):
    source_map = {
        'The Indian Express': 'https://news.google.com/rss/search?q=source:The+Indian+Express+election&hl=en-IN&gl=IN&ceid=IN:en',
        'Times of India': 'https://news.google.com/rss/search?q=source:The+Times+of+India+election&hl=en-IN&gl=IN&ceid=IN:en'
    }
    feed_url = source_map.get(source, '')
    if not feed_url:
        return "News source URL not found", 404

    # Get the specified number of news articles from the RSS feed
    articles = get_news_articles(feed_url, num_articles=100)

    return render_template('display_news.html', articles=articles, source=source)



@app.route('/analyze')
def analyze():
    url = request.args.get('url')
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Attempt to determine the source by checking meta tags or known content structures
    meta_publisher = soup.find("meta", property="og:site_name")
    publisher = meta_publisher["content"].lower() if meta_publisher else ""

    # Set selectors based on the publisher identified by meta tags
    if 'times of india' in publisher:
        content_blocks = soup.find_all('div', class_="_s30J clearfix")
    elif 'indian express' in publisher:
        content_blocks = soup.find_all('div', class_="story_details")
    else:
        return render_template('error.html', message="Unsupported news source.")

    article_content = ' '.join(block.text.strip() for block in content_blocks if block.text.strip() != '')

    if not article_content:
        return render_template('error.html', message="No content found to analyze.")

    try:
        preprocessed_text = preprocess(article_content)
        preprocessed_text_str = ' '.join(preprocessed_text)  # Convert list of words to string
        # Store preprocessed text in session for later use
        session['preprocessed_text'] = preprocessed_text_str
        # Generate word cloud
        text_combined = ' '.join(preprocessed_text)
        session['text_combined'] = text_combined
        wordcloud = WordCloud(max_font_size=50, max_words=90, background_color="white").generate(text_combined)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        session['wordcloud_img'] = image_base64

        # # Initialize and fit the CombinedTM model with loss weights
        # qt = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")
        # training_dataset = qt.fit(text_for_contextual=[preprocessed_text], text_for_bow=[preprocessed_text], labels=[0])
        # ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=50, loss_weights={"beta": 3})
        # ctm.fit(training_dataset)

        # # Get topics from the trained model
        # topics = ctm.get_topics(2)


        return render_template('analysis_news.html', article_content=article_content, preprocessed_text=preprocessed_text_str, wordcloud_img=image_base64)
    except Exception as e:
        return render_template('error.html', message="Failed to generate word cloud. Error: {}".format(str(e)))



@app.route('/save_news', methods=['POST'])
def save_news():
    preprocessed_text_str = session.get('preprocessed_text')
    if not preprocessed_text_str:
        return "No preprocessed data to download", 400

    from io import StringIO
    import csv
    from flask import Response

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Preprocessed Text'])
    # Assuming you want to save the entire preprocessed text in one cell
    writer.writerow([preprocessed_text_str])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=preprocessed_news_data.csv"}
    )


@app.route('/download_preprocessed_data')
def download_preprocessed_data():
    text_combined = session.get('text_combined')
    print(text_combined)
    if not text_combined:
        return "No preprocessed data found", 404

    output = StringIO()
    writer = csv.writer(output)
    words = text_combined.split()
    for i in range(0, len(words), 50):  # Split into chunks of 50 words
        writer.writerow(words[i:i+50])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=preprocessed_data.csv"}
    )




@app.route('/download_wordcloud')
def download_wordcloud():
    wordcloud_img = session.get('wordcloud_img')
    if not wordcloud_img:
        return "No wordcloud image found", 404

    # Decode the base64 encoded image
    img_data = base64.b64decode(wordcloud_img)

    # Prepare response to send the WordCloud image for download
    return Response(
        img_data,
        mimetype="image/png",
        headers={"Content-Disposition": "attachment;filename=wordcloud.png"}
    )


if __name__ == "__main__":
    app.run(debug=True)
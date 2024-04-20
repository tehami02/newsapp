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

@app.route('/display_news/<source>')
def display_news(source):
    source_map = {
        'The Indian Express': 'https://news.google.com/rss/search?q=source:"The+Indian+Express"',
        'Times of India': 'https://news.google.com/rss/search?q=source:"The+Times+of+India"'
    }
    feed_url = source_map.get(source, '')
    if not feed_url:
        return "News source URL not found", 404

    # Fetch news articles from RSS feed
    feed = feedparser.parse(feed_url)
    articles = feed.entries[:35]  # Get top news articles
    # session['articles'] = [{'title': entry.title, 'link': entry.link} for entry in articles]

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

        # Generate word cloud
        text_combined = ' '.join(preprocessed_text)
        session['text_combined'] = text_combined
        # print("Text for word cloud:", preprocessed_text)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_combined)
        # Create a new figure for plotting
        plt.figure()

        # Plot the word cloud image
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # Save the plot to a buffer in memory
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode the image to base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Store in session
        session['wordcloud_img'] = image_base64

        return render_template('analysis_news.html', article_content=article_content, preprocessed_text=preprocessed_text, wordcloud_img=image_base64)
    except Exception as e:
        return render_template('error.html', message="Failed to generate word cloud. Error: {}".format(str(e)))


@app.route('/save_news', methods=['POST'])
def save_news():
    headlines = request.form.getlist('headlines')
    urls = request.form.getlist('urls')

    if not headlines or not urls:
        return "No data to download", 400

    from io import StringIO
    import csv
    from flask import Response

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Headline', 'URL'])
    for headline, url in zip(headlines, urls):
        writer.writerow([headline, url])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=news_headlines.csv"}
    )



@app.route('/download_preprocessed_data')
def download_preprocessed_data():
    text_combined = session.get('text_combined', None)
    if not text_combined:
        return "No preprocessed data found", 404

    output = StringIO()
    writer = csv.writer(output)
    # Split the text_combined back into words for individual rows, if needed
    for word in text_combined.split():
        writer.writerow([word])  # Each word in its own row

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


# NLPC
Natural Language Processing Camp

![linkedin_cover_1](https://user-images.githubusercontent.com/66115008/119191867-e1686a80-ba87-11eb-8915-3e22180edbc0.png)

This GitHub repository contains all the "Paza Anulare" team's work throughout the Natural Language Processing Camp.

This project represents a telegram bot that uses Natural Language Processing to analyze the customers' review of a product using tweets from the online platform Tweeter and reddits from the online platform Reddit. This bot takes a product's name as an input and gives general feedback on the product based on customers' opinion from Twitter and Reddit.

There project contains three important modules in this project:

1)parse;

2)nets;

3)telegram-bot;

The general purpose of the files:

In the parse directory, you will find the code that focuses on parsing and collecting data from the online's platform. Here are 2 classes that realise it.

In the nets directory, you will find the nets that concentrate on the analysis of the data and their pipelines. We talk about Sentiment Analysis neural network, Gender Recognition neural network and an Api to gpt-3 model of the Openai Elon Musk's company that provide chat discussions.

In the telegram-bot directory, you will find the code that focuses on the bot-processes; also here, we combine all the components and we start the processes of parsing-analysis-answer.

Starting the code:

To start all the process works, you need to compile the telegram_bot.py file in the telegram-bot directory

Needed Libraries:

Afterwards, please make sure that you have all the needed libraries to compile the code. This is a list of all the libraries used in our project and the installation command for them (for those of you who don't know, if you have already installed python on your computer, then open Command Prompt and type the commands after the vertical bar to install the libraries):

1. nltk | pip install nltk
2. aiogram | pip install aiogram
3. pandas | pip install pandas
4. datetime | pip install DateTime
5. requests | pip install requests
6. bs4 | pip install bs4
7. scikit-learn | pip install scikit-learn
8. NumPy | pip install numpy
9. Matplotlib | pip install matplotlib
10. Seaborn | pip install seaborn
11. torch | pip install torch
12. torchtext | pip install torchtext
13. tweepy | pip install tweepy
14. praw | pip install praw
15. openai | pip install openai


You are ready to compile the code, so please feel free to run the telegram_bot.py file. To interact with the telegram bot and to get the results from it, you will probably need the link to the telegram chatbot, so here it is:

[http://t.me/PazaAnulare_NLP_Project_bot](http://t.me/PazaAnulare_NLP_Project_bot)

After compiling the code, to start the process, we recommend you use the function "/start" to start the chatbot interaction, but here are all the chatbot possibilities that we would like you to try:

# The possiblities of the chatbot

To analyze a particular product, type ”/analyze *your product*”.Consequently, you will get short statistics about your product, which includes: the number of comments, the number of positive comments; the number of negative comments; the positive per cent of the comments.

Important: Keep in mind that the analysis process may take some time, so don't worry about it. Usually, it takes about 25 seconds to process your input.

Once you've done the analyzing part you can also get some more information about your result. Also under the information message you will get some buttons for gender, trend, top, alternatives, definition and export data. This are their role:

Alternatives button: Gives you some alternatives for your product, and what people also ask about this product.

Gender button: Sends some diagrams about Number of comments, Number of comments for men and women, Men's positivity percentage, Women's positivity percentage.

![Nr_comments](https://user-images.githubusercontent.com/66115008/119814638-4cd09300-bef3-11eb-8d52-0750f5646f96.png)
![Nr_gender_comments](https://user-images.githubusercontent.com/66115008/119814752-6e317f00-bef3-11eb-8bca-eefe81599bc9.png)
![Nr_men_comments](https://user-images.githubusercontent.com/66115008/119814720-6540ad80-bef3-11eb-9308-6210bc9cc02c.png)
![Nr_woman_comments](https://user-images.githubusercontent.com/66115008/119814736-68d43480-bef3-11eb-9550-b22ad46f872c.png)

Top button: Sends a text message which includes the top 3 comments both positive and negative.

Definition button: Sends you the definition of the product.

Export data button: Sends you a csv with the analysis result

Also, we are creating more user-friendly output data for you, such as graphs and diagrams, so sooner enough, we will add new features to the code, so keep in mind to constantly check our repository for the latest updates.

Normal interaction part:

Our bot is also capable of everyday interaction such as finding the weather in different cities or telling jokes.

Overall, these are all the possibilities of our telegram bot, but we are looking forward to developing new features, so keep in mind to check our repository later.

Btw, this is our team that created this project:
![179208541_290533859189418_8294782364621965920_n](https://user-images.githubusercontent.com/66115008/119192167-5a67c200-ba88-11eb-84bd-2e28e7d32254.jpg)


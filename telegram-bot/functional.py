import json

from parse.reddit import RedditParser
from parse.twitter import TwitterParser

from nets.sentiment_analysis import SentimentPipeline
from nets.gender import GenderPipeline
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
import requests
import shutil
import os


class Parser:
    def __init__(self):
        self.twitterParser = TwitterParser()
        self.redditParser = RedditParser()

    def parse(self, user_input):
        df_twitter = self.twitterParser.get_posts(user_input, limit=20)
        df_reddit = self.redditParser.comments(user_input, limit=20)
        return df_twitter, df_reddit


class Statistics:
    def __init__(self):
        self.parser = Parser()
        self.sentiment_analyzer = SentimentPipeline()
        self.gender_analyzer = GenderPipeline()
        self.statistics = {}

    def __general_info(self, df: pd.DataFrame, user_input):
        self.statistics['theme'] = user_input
        self.statistics['nr_comments'] = df.shape[0]
        self.statistics['nr_positives'] = len(df[df.sentiment == 'Positive'])
        self.statistics['nr_negatives'] = len(df[df.sentiment == 'Negative'])
        self.statistics['nr_neutrals'] = len(df[df.sentiment == 'Neutral'])

        self.statistics['nr_male'] = len(df[df.gender == 'Male'])
        self.statistics['nr_female'] = len(df[df.gender == 'Female'])

        self.statistics['nr_positives_male'] = len(df[(df.sentiment == 'Positive') & (df.gender == 'Male')])
        self.statistics['nr_positives_female'] = len(df[(df.sentiment == 'Positive') & (df.gender == 'Female')])

        self.statistics['nr_negatives_male'] = len(df[(df.sentiment == 'Negative') & (df.gender == 'Male')])
        self.statistics['nr_negatives_female'] = len(df[(df.sentiment == 'Negative') & (df.gender == 'Female')])

        self.statistics['nr_neutrals_male'] = len(df[(df.sentiment == 'Neutral') & (df.gender == 'Male')])
        self.statistics['nr_neutrals_female'] = len(df[(df.sentiment == 'Neutral') & (df.gender == 'Female')])

        self.statistics['reddit_nr_positives'] = len(df[(df.source == 'reddit') & (df.sentiment == 'Positive')])
        self.statistics['reddit_nr_negatives'] = len(df[(df.source == 'reddit') & (df.sentiment == 'Negative')])
        self.statistics['reddit_nr_neutrals'] = len(df[(df.source == 'reddit') & (df.sentiment == 'Neutral')])

        self.statistics['reddit_nr_male'] = len(df[(df.source == 'reddit') & (df.sentiment == 'Male')])
        self.statistics['reddit_nr_female'] = len(df[(df.source == 'reddit') & (df.sentiment == 'Female')])

        self.statistics['twitter_nr_positives'] = len(df[(df.source == 'twitter') & (df.sentiment == 'Positive')])
        self.statistics['twitter_nr_negatives'] = len(df[(df.source == 'twitter') & (df.sentiment == 'Negative')])
        self.statistics['twitter_nr_neutrals'] = len(df[(df.source == 'twitter') & (df.sentiment == 'Neutral')])

        self.statistics['twitter_nr_male'] = len(df[(df.source == 'twitter') & (df.sentiment == 'Male')])
        self.statistics['twitter_nr_female'] = len(df[(df.source == 'twitter') & (df.sentiment == 'Female')])

    def __top_comments(self, df: pd.DataFrame):
        self.statistics['top_negative_comments'] = list(df.sort_values(by=['sentiment_value']).text[:3])
        self.statistics['top_positive_comments'] = list(
            df.sort_values(by=['sentiment_value'], ascending=False).text[:3])

    def __plot(self, path):
        # Creating a png named Nr_men_comments.pn that contains a plot with the number of positive and negative men's comments
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        labels = ['Number of men\'s positive comments', 'Number of men\'s negative comments']
        data = [self.statistics['nr_positives_male'], self.statistics['nr_positives_male']]
        ax.set_title('Men\'s total comments', fontsize=25)
        ax.pie(data, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 20})
        plt.savefig(path + r'\Nr_men_comments.png',
                    bbox_inches='tight')

        # Creating a png named Nr_comments.png which contains a plot with the number of positive and negative comments
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        labels = ['Number of positive comments', 'Number of negative comments']
        data = [self.statistics['nr_positives'], self.statistics['nr_negatives']]
        ax.set_title('Number of total comments', fontsize=25)
        ax.pie(data, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 20})
        plt.savefig(path + r'\Nr_comments.png',
                    bbox_inches='tight')

        # Creating a png named Nr_gender_comments.png which contains a plot with the gender of the users who wrote the comments
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        labels = ['Number of men\'s comments', 'Number of women\'s comments']
        data = [self.statistics['nr_male'], self.statistics['nr_female']]
        ax.set_title('Number of total comments', fontsize=25)
        ax.pie(data, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 20})
        plt.savefig(path + r'\Nr_gender_comments.png',
                    bbox_inches='tight')

        # Creating a png named Nr_woman_comments.png which contains a plot with the number of positive and negative women's comments
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        labels = ['Number of women\'s positive comments', 'Number of women\'s negative comments']
        data = [self.statistics['nr_positives_female'], self.statistics['nr_negatives_female']]
        ax.set_title('Women\'s total comments', fontsize=25)
        ax.pie(data, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 20})
        plt.savefig(path + r'\Nr_woman_comments.png',
                    bbox_inches='tight')

    def __definition(self, input_text):
        # takes the word as input
        search_word = input_text
        url = f'https://www.google.com/search?q={search_word}&hl=en'

        # Fetch the URL data using requests.get(url),
        # store it in a variable, request_result.
        request_result = requests.get(url)

        # Creating soup from the fetched request
        soup = BeautifulSoup(request_result.text, "html.parser")

        # Get the whole body tag
        tag = soup.body
        collected_strings = []

        # Filteres the information
        for string in tag.strings:
            collected_strings.append(string)

        definition = "Sorry, we could not find any definition for you."

        # Verifies if there is a wikipedia definition on google
        if ('Wikipedia') in collected_strings:
            alt_index = collected_strings.index('Wikipedia')
            definition = collected_strings[alt_index - 1]

        alternatives = ''
        if ('People also search for') in collected_strings:
            alt_index = collected_strings.index('People also search for')
            alternatives = collected_strings[alt_index + 1:alt_index + 7]
            if ("People also ask") in collected_strings:
                alternatives = ';\n '.join(collected_strings[
                                           collected_strings.index("People also ask"):collected_strings.index(
                                               "People also ask") + 5])
                alternatives = 'Alternatives:\n ' + ';\n '.join(
                    collected_strings[alt_index + 1:alt_index + 7]) + '\n\n' + alternatives
        else:
            alternatives = "Sorry, we could not find any alternatives for you."

        self.statistics['definition'] = definition
        self.statistics['alternatives'] = alternatives

    def clear_temp(self, path):
        try:
            shutil.rmtree(path)
        except:
            print('We can\'t delete. This path does not exist')

    def analyse(self, user_input: str, user_id):
        user_id = str(user_id)
        path = self.get_path(user_id)
        self.clear_temp(path)
        os.mkdir(path)

        df_twitter, df_reddit = self.parser.parse(user_input)
        df_twitter = self.gender_analyzer(self.sentiment_analyzer(df_twitter))
        df_reddit = self.gender_analyzer(self.sentiment_analyzer(df_reddit))
        df_twitter['source'] = 'twitter'
        df_reddit['source'] = 'reddit'
        df = pd.concat([df_twitter, df_reddit])

        self.__general_info(df, user_input)
        self.__top_comments(df)
        self.__plot(path)
        self.__definition(user_input)

        json.dump(self.statistics, open(path + os.sep + r'statistics.json', 'w'))
        df.to_csv(path + os.sep + r'data.csv')

    def get_path(self, user_id):
        return os.getcwd() + os.sep + r'temp_files' + os.sep + str(user_id)

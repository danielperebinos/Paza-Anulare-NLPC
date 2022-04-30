import tweepy
import pandas as pd

class TwitterParser:
    def __init__(self):
        self.__consumer_key = 'SNM41jzaukOPZKvpq7tyydTt3'
        self.__consumer_secret = 'uYs21vrjkSccpOgB9k0lI7RkwQJhOSrfsb3hiQLyX4anzGwDv2'
        self.__bearer_token = 'AAAAAAAAAAAAAAAAAAAAANVnawEAAAAAPXh3AKPskEbyiGa3Fc5Gjo5O%2FP4%3DeCw1AfHomEUnbSunkcLzecPZrM2SFnQpKQqT9LlrXNE1Hv1fhu'
        self.__access_token = '1508314112283009030-zdqccwsaFfFfn0oiAxaLkKAiJmwlBU'
        self.__access_token_secret = 'ySP9633PBwrN9ZsSRNhZMvpKUl61E99pCzI8xjJUpcwke'
        self.client = tweepy.Client(bearer_token=self.__bearer_token,
                       consumer_key=self.__consumer_key,
                       consumer_secret=self.__consumer_secret,
                       access_token=self.__access_token,
                       access_token_secret=self.__access_token_secret)


    def get_posts(self, theme, limit = 100):
        expansions = ['author_id','referenced_tweets.id','referenced_tweets.id.author_id','entities.mentions.username','attachments.poll_ids','attachments.media_keys','in_reply_to_user_id','geo.place_id']
        tweets = self.client.search_recent_tweets(query=theme,
                                             max_results=limit if limit > 0 and limit < 100 else 100,
                                             expansions= expansions,
                                             tweet_fields = 'geo')

        text = [str(tweet.text) for tweet in tweets.data]
        authors_id = [tweet.author_id for tweet in tweets.data]
        users = self.client.get_users(ids=authors_id)
        names = [str(user.name) for user in users.data]

        return pd.DataFrame({'name':names, 'text':text}).dropna()


if __name__ == '__main__':
    parser = TwitterParser()
    df = parser.get_posts('cats')
    print(df)
    df.to_csv('../example/twitter.csv')
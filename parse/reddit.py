import pandas as pd
import praw

class RedditParser():
    def __init__(self):
        self.__reddit = praw.Reddit(
            client_id="S9fooW9-b624C_-vtAx6jQ",
            client_secret="-_TLuJ1Im0vmkQTptc-LayTKOVLP6Q",
            username="__refjs__",
            password="telegrambot",
            user_agent="win64:cmt_scraper:v0.1 (u/TrAWei09)"
        )

    def posts(self, theme, limit = 100):
        subreddit = self.__reddit.subreddit(theme)
        submissions = subreddit.top(limit=limit)
        return list(map(lambda x: x.title, submissions))

    def comments(self, theme, limit = 1):
        subreddit = self.__reddit.subreddit(theme)
        submissions = subreddit.top(limit=limit)
        names = []
        text = []

        for submission in submissions:
            comments = submission.comments
            for comment in comments:
                try:
                    names.append(str(comment.author))
                    text.append(str(comment.body) if comment.body != '[deleted]' else None)
                    if len(comment.replies) > 0:
                        for reply in comment.replies:
                            names.append(str(reply.author))
                            text.append(str(reply.body) if reply.body != '[deleted]' else None)
                except:
                    print(comment)

        return pd.DataFrame({'name': names, 'text':text}).dropna()


if __name__ == '__main__':
    parser = RedditParser()
    df = parser.comments('cats', limit = 10)
    print(df)
    df.to_csv('../example/reddit.csv')
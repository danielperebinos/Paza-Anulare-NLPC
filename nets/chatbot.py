import openai

class gptbot:
    def __init__(self):
        self.key = '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'

    def response(self, text:str):
        openai.api_key = self.key
        response = openai.Completion.create(
            # engine = 'ada'
            engine = 'text-davinci-002',
            prompt = text,
            temperature = 0.5,
            max_tokens = 100,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )

        content = response.choices[0].text.split(sep='.')[0]
        print(content)
        return content



if __name__ == '__main__':
    bot = gptbot()
    bot.response('Hello my friend!')
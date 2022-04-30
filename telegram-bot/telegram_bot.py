"""
 Author: Perebinos Daniel
"""

import logging
import os

from aiogram import Bot, Dispatcher, executor, types
from markups import main_menu, call_menu, more_details
from functional import Statistics
from nets.chatbot import gptbot
import json


# API_TOKEN = '5173248256:AAE0dMbydHDbVSBwZb0xR5-gdpV172LS8P0'
API_TOKEN = '1759313825:AAGevPY6RL6oqWyOkAZpQJF1B1iBzTn4VG4'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
statistics = Statistics()
openai = gptbot()

# Message Handlers
@dp.message_handler(commands=['start', 'Start'])
async def start(message: types.Message):
    """
    This handler will be called when user sends /start or /help command
    """
    await message.reply("Hi!\nI'm Paza-Anulare Bot!\nI'm a tool for analyse business objects, but i'm also can"
                        "be your friend.", reply_markup=call_menu)

@dp.message_handler(commands=['menu', 'Menu'])
async def get_menu(message: types.Message):
    """
    This handler will be called when user sends /menu or /Menu command
    """
    await bot.send_message(message.from_user.id, 'The menu is: ', reply_markup=main_menu)

@dp.message_handler(commands=['help', 'Help'])
async def help(message: types.Message):
    text = "IMPORTANT: \n\n To use all the funcitons like gender, trends or alternative you first need to " \
           "analyze the product in this form: '/analyze *your product*' \n\n\nYou can use this commands : " \
           "\n\n/info - short information about the project\n\n/help - displays all the functions that we " \
           "have\n\n/menu - gives the menu for more detailed result\n\n/analyse - function needed to start " \
           "the analysis process|'/analyse *your product*'\n\n/top - gives the top positive and negative " \
           "comments\n\n/gender - gives some diagrams about gender\n\n/alternative - gives some alternatives for " \
           "your product if existing\n\n/definition - gives the definition of the product if existing\n\n" \
           "/export - sends you the Analysis result in a *.csv form\n\n"
    await bot.send_message(message.from_user.id, text)
    await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)

@dp.message_handler(commands=['analyse', 'Analyse'])
async def analyse(message: types.Message):
    """
    This handler will be called when user sends /analyse or /Analyse command
    """
    text = message.text[len('/analyse '):]

    await bot.send_message(message.from_user.id, 'Analyse: ' + text)
    await bot.send_message(message.from_user.id,'It might take some time')

    user_id = str(message.from_user.id)
    statistics.analyse(text, user_id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json', 'r'))

    await bot.send_message(message.from_user.id,
        f"General Info\nNumber of comments: {data['nr_comments']}\n"
        f"Number of the positive comments: {data['nr_positives']} ({round(data['nr_positives']/data['nr_comments'],2)}%)\n"
        f"Number of the negative comments: {data['nr_negatives']} ({round(data['nr_negatives']/data['nr_comments'],2)}%)\n"
        f"Number of the neutral comments: {data['nr_neutrals']} ({round(data['nr_neutrals'] / data['nr_comments'],2)}%)\n",
        reply_markup=more_details
    )
    await bot.send_message(message.from_user.id,'You can also try: ', reply_markup=main_menu)

@dp.message_handler(commands=['top', 'Top'])
async def top(message: types.Message):
    """
    This handler will be called when user sends /top or /Top command
    """
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        data = json.load(open(path + os.sep + 'statistics.json', 'r'))
        await bot.send_message(message.from_user.id,
                               f"Top negative comments:\n1. {data['top_negative_comments'][0]}\n"
                               f"2. {data['top_negative_comments'][1]}\n3. {data['top_negative_comments'][2]}\n\n"
                               f"Top positive comments:\n1. {data['top_positive_comments'][0]}\n"
                               f"2. {data['top_positive_comments'][1]}\n3. {data['top_positive_comments'][2]}"
       )
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)

@dp.message_handler(commands=['gender', 'Gender'])
async def gender(message: types.Message):
    """
    This handler will be called when user sends /gender or /Gender command
    """
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_gender_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_men_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_woman_comments.png', 'rb'))
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)

@dp.message_handler(commands=['alternative', 'Alternative'])
async def alternative(message: types.Message):
    """
    This handler will be called when user sends /alternative or /Alternative command
    """
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json'))
    if os.path.exists(path):
        await bot.send_message(user_id, data['alternatives'])
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)

@dp.message_handler(commands=['definition', 'Definition'])
async def definition(message: types.Message):
    """
    This handler will be called when user sends /definition or /Definition command
    """
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json'))
    if os.path.exists(path):
        await bot.send_message(user_id, data['definition'])
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)

@dp.message_handler(commands=['export', 'Export'])
async def export(message: types.Message):
    """
    This handler will be called when user sends /export or /Export command
    """
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        await bot.send_document(user_id, open(path + os.sep + 'data.csv', 'rb'))
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)

@dp.message_handler()
async def chat(message: types.Message):
    """
    This handler will be called when user sends /top or /Top command
    """
    await message.reply(openai.response(str(message.text)))


# Callback query handlers
@dp.callback_query_handler(text='menu')
async def menu(message: types.Message):
    await bot.send_message(message.from_user.id, text= 'The Menu is:', reply_markup=main_menu)

@dp.callback_query_handler(text='top')
async def top(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        data = json.load(open(path + os.sep + 'statistics.json', 'r'))
        await bot.send_message(message.from_user.id,
                               f"Top negative comments:\n1. {data['top_negative_comments'][0]}\n"
                               f"2. {data['top_negative_comments'][1]}\n3. {data['top_negative_comments'][2]}\n\n"
                               f"Top positive comments:\n1. {data['top_positive_comments'][0]}\n"
                               f"2. {data['top_positive_comments'][1]}\n3. {data['top_positive_comments'][2]}"
                               )
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await bot.send_message(message.from_user.id, 'You have not realised analysis', reply_markup=call_menu)

@dp.callback_query_handler(text='gender')
async def gender(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_gender_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_men_comments.png', 'rb'))
        await bot.send_photo(message.from_user.id, open(path + os.sep + 'Nr_woman_comments.png', 'rb'))
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await bot.send_message(message.from_user.id, 'You have not realised analysis', reply_markup=call_menu)

@dp.callback_query_handler(text='alternative')
async def alternative(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json'))
    if os.path.exists(path):
        await bot.send_message(user_id, data['alternatives'])
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await bot.send_message(message.from_user.id, 'You have not realised analysis', reply_markup=call_menu)

@dp.callback_query_handler(text='definition')
async def definition(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json'))
    if os.path.exists(path):
        await bot.send_message(user_id, data['definition'])
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await bot.send_message(message.from_user.id, 'You have not realised analysis', reply_markup=call_menu)

@dp.callback_query_handler(text='export')
async def export(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    if os.path.exists(path):
        await bot.send_document(user_id, open(path + os.sep + 'data.csv', 'rb'))
        await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)
    else:
        await bot.send_message(message.from_user.id, 'You have not realised analysis', reply_markup=call_menu)

@dp.callback_query_handler(text='help')
async def help(message: types.Message):
    text = "IMPORTANT: \n\n To use all the funcitons like gender, trends or alternative you first need to " \
           "analyze the product in this form: '/analyze *your product*' \n\n\nYou can use this commands : " \
           "\n\n/info - short information about the project\n\n/help - displays all the functions that we " \
           "have\n\n/menu - gives the menu for more detailed result\n\n/analyse - function needed to start " \
           "the analysis process|'/analyse *your product*'\n\n/top - gives the top positive and negative " \
           "comments\n\n/gender - gives some diagrams about gender\n\n/alternative - gives some alternatives for " \
           "your product if existing\n\n/definition - gives the definition of the product if existing\n\n" \
           "/export - sends you the Analysis result in a *.csv form\n\n"
    await bot.send_message(message.from_user.id, text)
    await bot.send_message(message.from_user.id,'You can try: ', reply_markup=main_menu)

@dp.callback_query_handler(text='details')
async def details(message: types.Message):
    user_id = str(message.from_user.id)
    path = statistics.get_path(user_id)
    data = json.load(open(path + os.sep + 'statistics.json'))
    if os.path.exists(path):
        text = 'Details:\n'
        for key, value in data.items():
            if key not in ['top_negative_comments', 'top_positive_comments']:
                text += str(key) + ': ' + str(value) + '\n'
        await bot.send_message(message.from_user.id, text)
        await bot.send_message(message.from_user.id, 'You can try: ', reply_markup=main_menu)
    else:
        await message.reply('You have not realised analysis', reply_markup=call_menu)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
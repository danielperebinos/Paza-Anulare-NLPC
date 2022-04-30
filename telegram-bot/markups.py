from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

main_menu = InlineKeyboardMarkup(row_width=2)

help_button = InlineKeyboardButton(text = 'Help', callback_data='help')
top_button = InlineKeyboardButton(text = 'Top', callback_data='top')
gender_button = InlineKeyboardButton(text = 'Gender', callback_data='gender')
alternative_button = InlineKeyboardButton(text = 'Alternative', callback_data='alternative')
definition_button = InlineKeyboardButton(text = 'Definition', callback_data='definition')
export_button = InlineKeyboardButton(text = 'Export', callback_data='export')

main_menu.insert(help_button)
main_menu.insert(top_button)
main_menu.insert(gender_button)
main_menu.insert(alternative_button)
main_menu.insert(definition_button)
main_menu.insert(export_button)



call_menu = InlineKeyboardMarkup(row_width=2)
menu = InlineKeyboardButton(text='Menu', callback_data='menu')
call_menu.insert(menu)

more_details = InlineKeyboardMarkup(row_width=2)
details = InlineKeyboardButton(text='Details', callback_data='details')
more_details.insert(details)
import asyncio
import signal
from telegram import Bot, Update
from telegram.ext import Updater


async def process_updates(queue: asyncio.Queue, bot: Bot, command_queue: asyncio.Queue, my_chat_id):
    """Process incoming updates and print information about the sender of each message."""
    while True:
        update = await queue.get()
        if isinstance(update, Update) and update.message:
            message = update.message
            sender = message.from_user  # User object representing the sender
            chat_id = message.chat_id
            incoming_text = message.text

            # Extract sender details
            sender_id = sender.id
            sender_username = sender.username  # Could be None if the user has no username set
            sender_name = f"{sender.first_name} {sender.last_name if sender.last_name else ''}".strip()

            # Print sender information and the message text

            if sender_id == int(my_chat_id):

                print(
                    f"Message from {sender_name} (@{sender_username if sender_username else 'No Username'}): {incoming_text}")

                # Optionally, send a response (e.g., echoing the received message and sender name)
                # response_text = f"Received your message, {sender_name}: '{incoming_text}'"
                # await bot.send_message(chat_id=chat_id, text=response_text)

                # Put the command into the command_queue
                incoming_text = extract_incoming_text(incoming_text)
                await command_queue.put((incoming_text, sender_id))

            queue.task_done()


async def trading_strategy(command_queue: asyncio.Queue, my_chat_id):
    """Your trading strategy implementation here."""
    while True:
        # Example of a trading strategy task
        print("Running trading strategy...")

        # Check for any commands or conditions to adjust the strategy
        try:
            # command is (str) and sender_id is (int)
            command, sender_id = await asyncio.wait_for(command_queue.get(), timeout=5)
            if sender_id == int(my_chat_id):
                print(f"Received command: {command}")
            # Process the command and adjust the strategy accordingly
            command_queue.task_done()
        except asyncio.TimeoutError:
            pass

def extract_incoming_text(text):
    # Initialize the command as an empty string
    incoming_text = ""

    # Check if the input is not None and is a command (starts with '/')
    if text is not None and text.startswith('/'):
        # Check if the command contains the bot's username with '@'
        if '@' in text:
            # Split the text at '@' and take the first part
            command_with_slash = text.split('@')[0]
        else:
            command_with_slash = text

        # Ensure the command_with_slash has more than just '/'
        if len(command_with_slash) > 1:
            # Remove the '/' to extract the actual command
            incoming_text = command_with_slash[1:]

    return incoming_text


# async def main():
#     TOKEN = '6741244950:AAGl8erp5Kmh67prHEfj7rDHb3jH1THzXgI'
#     my_chat_id = '6473165102'
#     bot = Bot(token=TOKEN)
#     update_queue = asyncio.Queue()
#     command_queue = asyncio.Queue()
#     updater = Updater(bot=bot, update_queue=update_queue)
#
#     async with updater:
#         updater_task = asyncio.create_task(updater.start_polling())
#         processing_task = asyncio.create_task(process_updates(update_queue, bot, command_queue, my_chat_id))
#         trading_task = asyncio.create_task(trading_strategy(command_queue, my_chat_id))
#
#         try:
#             await asyncio.gather(updater_task, processing_task, trading_task)
#         except asyncio.CancelledError:
#             print("Tasks were cancelled.")
#         finally:
#             # Ensure this code runs after the tasks are cancelled
#             print("Executing code after the tasks...")


# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#     for sig in (signal.SIGINT, signal.SIGTERM):
#         loop.add_signal_handler(sig, loop.stop)
#     try:
#         loop.run_until_complete(main())
#     finally:
#         print("Shutting down...")

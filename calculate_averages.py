import os
from dotenv import load_dotenv
import utils 

if __name__ == "__main__":
    load_dotenv() # Load environment variables from .env file

    JSONBIN_API_KEY = os.environ.get('JSONBIN_API_KEY')
    JSONBIN_BIN_ID = os.environ.get('JSONBIN_BIN_ID')

    if not JSONBIN_API_KEY or not JSONBIN_BIN_ID:
        print("Error: JSONBIN_API_KEY or JSONBIN_BIN_ID not found in environment variables.")
        print("Make sure you have a .env file with these values.")
    else:
        utils.calculate_and_save_average_scores(api_key=JSONBIN_API_KEY, bin_id=JSONBIN_BIN_ID)

    print("Average score calculation finished.")
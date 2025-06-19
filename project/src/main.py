import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

import sys
from typing import Dict, Callable
from display.menu_handler import MenuHandler
from display.display_utils import DisplayUtils
from utils.get_user_inputs import GetUserInputs
from utils.colors import Colors

raw_file = pd.read_csv("../data/raw/online_gaming_behavior_dataset.csv")

print(raw_file.head(5))
print(raw_file.isnull().sum())

def main():
    """Fonction principale du launcher"""
    
    # Dictionnaire des actions disponibles
    menu_actions: Dict[str, Callable] = {
        '1': lambda: MenuHandler.handle_placeholder_option("About Data Cleaning"),
        '2': lambda: MenuHandler.handle_placeholder_option("About Preprocessing"),
        'C': MenuHandler.display_credits,
    }
    
    while True:
        DisplayUtils.clear_screen()
        MenuHandler.display_welcome()
        print()
        MenuHandler.display_menu()
        
        choice = GetUserInputs.get_user_choice()
        
        if choice == 'Q':
            print(f"\n{Colors.ORANGE}Thank you for trying our project !{Colors.RESET}")
            sys.exit(0)
        
        elif choice in menu_actions:
            menu_actions[choice]()
        
        else:
            print(f"\n{Colors.RED}❌ Invalid choice: '{choice}'{Colors.RESET}")
            print("Please select a valid option from the menu.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.ORANGE}Program interrupted by user.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}An error occurred: {e}{Colors.RESET}")
        sys.exit(1)
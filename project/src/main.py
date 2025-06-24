import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

import sys
from typing import Dict, Callable
from display.menu_handler import MenuHandler
from display.display_utils import DisplayUtils
from utils.get_user_inputs import GetUserInputs
from utils.colors import Colors
from utils.constants import *
from data.data_cleaning_handler import DataCleaningHandler
from data.data_preprocessing_handler import DataPreprocessingHandler
from visualization.labels_visualization import LabelVisualiation

from ai_generated import AI_Gen, RandomForestAnalyzer

raw_df = pd.read_csv("../data/raw/online_gaming_behavior_dataset.csv")

def main():
    """Fonction principale du launcher"""
    
    # Dictionnaire des actions disponibles
    menu_actions: Dict[str, Callable] = {
        '1': lambda: DataCleaningHandler.display_reflexion(raw=raw_df),
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
            DisplayUtils.clear_screen()
            menu_actions[choice]()
            break
        
        else:
            print(f"\n{Colors.RED}❌ Invalid choice: '{choice}'{Colors.RESET}")
            print("Please select a valid option from the menu.")
            input("\nPress Enter to continue...")

def currently(df: pd.DataFrame):
    # LabelVisualiation.display_labels_proportion_to_target(df=raw_df)
    df = DataPreprocessingHandler.preprocess_dataset(df=df)
    print(df.head(5))
    analyzer = RandomForestAnalyzer(df, target_column='EngagementLevel_High')
    results = analyzer.run_complete_analysis(n_estimators=200, top_features=10)

currently(raw_df)

# AI_Gen.en_cours(raw_df)

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print(f"\n\n{Colors.ORANGE}Program interrupted by user.{Colors.RESET}")
#         sys.exit(0)
#     except Exception as e:
#         print(f"\n{Colors.RED}An error occurred: {e}{Colors.RESET}")
#         sys.exit(1)g
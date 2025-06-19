from utils.colors import Colors
from display.display_utils import DisplayUtils

class MenuHandler :
    def display_welcome():
        """Affiche le message de bienvenue en ASCII art orange"""
        welcome_art = f"""{Colors.ORANGE}
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║                                                                            ║
        ║       ██╗    ██╗███████╗██╗      ██████╗ ██████╗ ███╗   ███╗███████╗       ║
        ║       ██║    ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗ ████║██╔════╝       ║
        ║       ██║ █╗ ██║█████╗  ██║     ██║     ██║   ██║██╔████╔██║█████╗         ║
        ║       ██║███╗██║██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║██╔══╝         ║
        ║       ╚███╔███╔╝███████╗███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║███████        ║
        ║        ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════        ║
        ║                                                                            ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        {Colors.RESET}"""
        
        print(welcome_art)


    def display_menu():
        """Affiche le menu en ASCII art avec question normale"""
        menu_frame = f"""{Colors.CYAN}
        ┌────────────────────────────────────────────────────────────────────────────┐
        │                                                                            │
        │  TABLE OF CONTENTS                                                         │
        │                                                                            │
        │  [1] About Data Cleaning                                                   │
        │  [2] About Preprocessing                                                   │
        │                                                                            │
        │  [C] Credits & Information                                                 │
        │  [Q] Quit                                                                  │
        │                                                                            │
        └────────────────────────────────────────────────────────────────────────────┘
        {Colors.RESET}"""
        
        print(menu_frame)
        print("What do you want to do?")


    def display_credits():
        DisplayUtils.clear_screen()
        
        print(f"{Colors.ORANGE}=== CREDITS & INFORMATION ==={Colors.RESET}")
        print()
        print(f"{Colors.BOLD}PROJECT:{Colors.RESET} Machine Learning on predicting online gaming behavior")
        print(f"{Colors.BOLD}GROUP:{Colors.RESET} 6")
        print()
        print(f"{Colors.BOLD}GROUP MEMBERS:{Colors.RESET}")
        print("• Membre 1: Esteban")
        print("• Membre 2: Grégoire") 
        print("• Membre 3: Alystan")
        print()
        print(f"{Colors.BOLD}SURPEVISION:{Colors.RESET}")
        print("• KMUTT's teachers")
        print()
        print(f"{Colors.BOLD}DATASET INFORMATION:{Colors.RESET}")
        print("• Owner: Rabie El Kharoua")
        print("• Title: 🎮 Predict Online Gaming Behavior Dataset")
        print("• Platform: Kaggle") 
        print("• License: CC BY 4.0")
        print("• DOI: https://doi.org/10.34740/KAGGLE/DSV/8742674")
        print("• Year: 2024")
        print()
        print(f"{Colors.YELLOW}This dataset is made available under the CC BY 4.0 license.{Colors.RESET}")
        print(f"{Colors.YELLOW}This is an original synthetic dataset created for educational purposes.{Colors.RESET}")
        
        input(f"\nPress Enter to return to main menu...")


    def handle_placeholder_option(option_name: str):
        print(f"\n🚧 {option_name} - Coming Soon!")
        print("This feature will be implemented as the project progresses.")
        input("\nPress Enter to continue...")
import os

class DisplayUtils :
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
ENDC = '\033[0m'

def print_color(string, color, bold=False):
    """
    Formats the string with colors for terminal prints
    """
    if bold is True:
        print(BOLD + color + string + ENDC)
    else:
        print(color + string + ENDC)

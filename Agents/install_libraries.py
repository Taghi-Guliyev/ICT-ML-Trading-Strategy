# install_libraries.py
# Purpose: To automatically check for and install all required libraries
#          for the agentic trading system. This script ensures that the
#          user's environment is correctly set up to run all agents.

import subprocess
import sys

def install_package(package):
    """
    Installs a given package using pip.
    Args:
        package (str): The name of the package to install.
    """
    try:
        print(f"Installing {package}...")
        # The following command runs 'pip install' for the specified package.
        # It uses sys.executable to ensure it's using the pip associated
        # with the current Python interpreter.
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def check_and_install_libraries(libraries):
    """
    Checks if a list of libraries is installed. If a library is not
    found, it attempts to install it.
    Args:
        libraries (list): A list of strings, where each string is the
                          name of a library to check.
    """
    print("--- Checking for required libraries ---")
    for library in libraries:
        try:
            # We try to import the library to see if it's installed.
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            # If the import fails, the library is not installed.
            print(f"{library} not found.")
            install_package(library)
    print("\n--- Library check complete ---")

if __name__ == "__main__":
    # This is the list of all unique libraries used across all agent files.
    # Note: 'mplfinance' and 'TA-Lib' might require special handling
    # depending on the user's operating system.
    required_libraries = [
        "pandas",
        "numpy",
        "scipy",
        "ta",
        "joblib",
        "matplotlib",
        "seaborn",
        "sklearn",
        "mplfinance",
        "talib"
    ]

    check_and_install_libraries(required_libraries)

    # A special note for TA-Lib, which can be tricky to install.
    print("\n--- IMPORTANT NOTE for TA-Lib ---")
    print("If the 'talib' installation failed, you might need to install its dependencies manually.")
    print("Please refer to the official TA-Lib installation guide for your operating system:")
    print("https://github.com/mrjbq7/ta-lib")


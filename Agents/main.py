# main.py
# Purpose: To orchestrate the entire quantitative trading strategy workflow,
#          from data preparation and model training to final backtesting and analysis.
#          This script acts as the central conductor for all the agents.

import sys
import config # Your custom configuration file

# Import the main function from each agent
# We use 'if __name__ == "__main__"' blocks in the agent files so they don't
# run automatically when we import them here.
from build_training_dataset import main as build_dataset
from model_trainer import main as train_model
from main_run_backtest import main as run_backtest
from performance_analyzer import main as analyze_performance
from chart_visualizer import main as visualize_chart

def main_menu():
    """
    Displays the main menu and prompts the user for a choice.
    """
    print("\n" + "="*50)
    print("      Quantitative Trading Strategy Orchestrator")
    print("="*50)
    print("Please choose which part of the pipeline to run:")
    print("\n--- RESEARCH & TRAINING (Run these first) ---")
    print("1. Build Training Dataset (from 2023-2024 data)")
    print("2. Train RF Model (using the dataset from step 1)")
    
    print("\n--- OUT-OF-SAMPLE BACKTESTING ---")
    print("3. Run Strategy with ML Filter on New Data (2025 data)")
    
    print("\n--- ANALYSIS & VISUALIZATION ---")
    print("4. Run Full Performance Analysis (on results from step 3)")
    print("5. Visualize Last 7k Bars of Backtest (from results of step 3)")
    
    print("\n--- FULL PIPELINE ---")
    print("6. Run EVERYTHING in sequence (1 -> 2 -> 3 -> 4 -> 5)")
    
    print("\n0. Exit")
    print("="*50)
    
    while True:
        try:
            choice = int(input("Enter your choice (0-6): "))
            if 0 <= choice <= 6:
                return choice
            else:
                print("Invalid choice. Please enter a number between 0 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """
    Main conductor function to run the selected pipeline step.
    """
    
    while True:
        choice = main_menu()

        if choice == 1:
            print("\n>>> RUNNING STEP 1: Building Training Dataset...")
            build_dataset()
        elif choice == 2:
            print("\n>>> RUNNING STEP 2: Training Model...")
            train_model()
        elif choice == 3:
            print("\n>>> RUNNING STEP 3: Running Backtest on Unseen Data...")
            run_backtest()
        elif choice == 4:
            print("\n>>> RUNNING STEP 4: Running Performance Analysis...")
            analyze_performance()
        elif choice == 5:
            print("\n>>> RUNNING STEP 5: Visualizing Chart...")
            visualize_chart()
        elif choice == 6:
            print("\n>>> RUNNING FULL PIPELINE (1-5)...")
            try:
                print("\n>>> STEP 1: Building Training Dataset...")
                build_dataset()
                print("\n>>> STEP 2: Training Model...")
                train_model()
                print("\n>>> STEP 3: Running Backtest on Unseen Data...")
                run_backtest()
                print("\n>>> STEP 4: Running Performance Analysis...")
                analyze_performance()
                print("\n>>> STEP 5: Visualizing Chart...")
                visualize_chart()
                print("\n--- FULL PIPELINE COMPLETED SUCCESSFULLY ---")
            except Exception as e:
                print(f"\n--- An error occurred during the full pipeline run: {e} ---")
        elif choice == 0:
            print("Exiting.")
            sys.exit()

if __name__ == "__main__":
    main()
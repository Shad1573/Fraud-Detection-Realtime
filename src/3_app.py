import pandas as pd
import joblib
import time
import os
from colorama import Fore, Style, init

# Initialize colors for the terminal
init(autoreset=True)

def run_terminal_monitor():
    # 1. Load the Brain
    if not os.path.exists('models/fraud_model.pkl'):
        print(Fore.RED + "Error: Model not found. Run Day 2 training first!")
        return
    
    model = joblib.load('models/fraud_model.pkl')
    df = pd.read_csv('data/creditcard.csv', nrows=10000) # Load a subset
    
    print(Fore.CYAN + "="*50)
    print(Fore.CYAN + "      FRAUD-SHIELD: TERMINAL LIVE MONITOR")
    print(Fore.CYAN + "="*50)
    print("Press Ctrl+C to stop the feed...\n")
    time.sleep(2)

    try:
        while True:
            # Simulate a live transaction arriving
            sample = df.sample(1)
            features = sample.drop(['Class'], axis=1)
            amount = features['Amount'].values[0]
            
            # AI Prediction
            prob = model.predict_proba(features)[0][1]
            
            # Logic for alerts
            if prob > 0.5:
                print(Fore.RED + f" [!] ALERT: FRAUD DETECTED | Amount: ${amount:.2f} | Risk: {prob:.2%}")
                print(Fore.YELLOW + f" Details: {features.iloc[:, 1:5].values}") # Show some PCA features
            else:
                print(Fore.GREEN + f" [✓] SECURE: Transaction Clear | Amount: ${amount:.2f} | Risk: {prob:.4%}")
            
            time.sleep(0.8) # Adjust speed of the live feed here
            
    except KeyboardInterrupt:
        print(Fore.MAGENTA + "\nStopping Live Feed... System Shutting Down.")

if __name__ == "__main__":
    run_terminal_monitor()
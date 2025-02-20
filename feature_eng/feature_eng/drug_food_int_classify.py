import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging


#
# The script takes many hours.
#

client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
        )

# Logger configuration 
logger = logging.getLogger("classify_interaction")
logger.setLevel(logging.INFO)  # Set the log level

# StreamHandler configuration
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Adding handler
logger.addHandler(handler)


#
# Use GPT to classify the interaction
#
def classify_interaction(interaction):
    prompt = """Interaction between drugs and foods:  
{interaction}  

Pharmacological: [Chemical / Physical / Biochemical / None]
Pharmacokinetic: [Absorption / Distribution / Metabolism / Excretion / None]
Pharmacodynamic: [Additive / Synergistic / Antagonistic / None]
Severity: [Minor / Moderate / Major]
Timing: [Immediate / Delayed interactions / Cumulative]

Answer only with the requested classifications.

Pharmacological: 
Pharmacokinetic:
Pharmacodynamic:
Severity:
Timing:"""
    #msg = [{"role": "system", "content": self.configurator.system_prompt}]
    msg = []
    msg.append({"role": "user", "content":  prompt.format(interaction=interaction)})
    response = client.chat.completions.create(model="gpt-4o-mini",
                                            messages=msg,
                                            seed=123,
                                            max_tokens=2048,
                                            temperature = 0)
    cleaned_text = response.choices[0].message.content
    lines = cleaned_text.splitlines()
    res = {"Pharmacological":None, "Pharmacokinetic":None, "Pharmacodynamic": None, "Severity": None, "Timing":None}
    for l in lines:
        if (i := l.find("Pharmacological:")) !=-1:
            j = len("Pharmacological:")
            res["Pharmacological"] = l[i+j:].strip().lower()
        elif (i := l.find("Pharmacokinetic:")) !=-1:
            j = len("Pharmacokinetic:")
            res["Pharmacokinetic"] = l[i+j:].strip().lower()
            if res["Pharmacokinetic"] == "none":
                res["Pharmacokinetic"] = None
        elif (i:=l.find("Pharmacodynamic:"))!=-1:
            j = len("Pharmacodynamic:")
            res["Pharmacodynamic"] = l[i+j:].strip().lower()
            if res["Pharmacodynamic"] == "none":
                res["Pharmacodynamic"] = None
        elif (i:=l.find("Severity:")) !=-1:
            j = len("Severity:")
            res["Severity"] = l[i+j:].strip().lower()
        elif (i:=l.find("Timing:")) !=-1:    
            j = len("Timing:")
            res["Timing"] = l[i+j:].strip().lower()

    return res

#
# Main function to update interaction classification
#
def update_interaction_classification(csv_file, batch_size, wait_time):
    try:
        # Load the csv file containing interction descriptions
        df = pd.read_csv(csv_file)

        # Add the new columns if needed
        required_columns = ["pharmacological", "pharmacokinetic_effect",
                            "pharmacodynamic_effect", "severity", "timing"]
        
        for column in required_columns:
            if column not in df.columns:
                df[column] = None  # Add the column and initialize it with None

        # Single row update
        def process_row(index, row):
            #logger.debug(f"process_row {index}")
            return index, {"Pharmacological": row["pharmacological"], "Pharmacokinetic":row["pharmacokinetic_effect"],
                           "Pharmacodynamic":row["pharmacodynamic_effect"], "Severity": row["severity"],
                           "Timing":row["timing"]}

        # Split the Dataframe into batches of 4500 rows
        #batch_size = batch_size_a or 4500
        num_batches = (len(df) + batch_size - 1) // batch_size  # Compute the number of batches 

        with tqdm(total=len(df), desc="Processing Interactions", unit="row") as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                # Use ThreadPoolExecutor to perform batches 
                with ThreadPoolExecutor(max_workers=10) as executor:
                    #futures = {executor.submit(process_row, idx, row): idx for idx, row in batch_df.iterrows()}
                    futures = {}
                    cont = 0
                    for idx, row in batch_df.iterrows():
                        if pd.isna(row["severity"]):
                            future = executor.submit(process_row, idx, row)
                            futures[future] = idx
                        cont +=1
                    #logger.debug("Features submitted")
                    if not futures:
                        pbar.update(cont)
                        continue

                    for future in as_completed(futures):
                        index, classification = future.result()
                        
                        # Update the dataframe with the results
                        if classification:
                            df.at[index, "pharmacological"] = classification["Pharmacological"]
                            df.at[index, "pharmacokinetic_effect"] = classification["Pharmacokinetic"]
                            df.at[index, "pharmacodynamic_effect"] = classification["Pharmacodynamic"]
                            df.at[index, "severity"] = classification["Severity"]
                            df.at[index, "timing"] = classification["Timing"]
                        else:
                            raise ValueError("No classification Found")
                        
                        #logger.debug(f"Index {index}")
                        # Aggiorna la progress bar
                        pbar.update(1)

                # Wait 20 seconds before elaboration the next batch. 
                # It is a fallback to avoid OpenAI RPM limits
                if i < num_batches - 1:  # Do not wait after the last batch 
                    logger.info(f"Waiting {wait_time} seconds before elaborating the next batch...")
                    time.sleep(wait_time)

        # Save the updated CSV 
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Update completed. Results saved to {csv_file}")

    except Exception as e:
        # Error handling: Save the actual Dataframe 
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.warning(f"Error during the elaboration: {e}")
        logger.warning(f"Actual state saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update interaction classifications.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file to elaborate.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--batch_size', type=int, default=4500, help='Number rows for batch (default: 4500)')
    parser.add_argument('--wait_time', type=int, default=4500, help='Number of seconds to sleep after each batch for batch (default: 15 sec.)')

    # Analize command line switches
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    update_interaction_classification(args.csv_file, args.batch_size or 4500, args.wait_time or 15 )
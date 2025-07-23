import pandas as pd
import glob

file_list = glob.glob("Fp2_20*_data.csv") 

df_list = []
for file in file_list:
    df = pd.read_csv(file)
    df_list.append(df)

combined_fp2 = pd.concat(df_list, ignore_index=True)

combined_fp2 = combined_fp2.sort_values(by=["Year", "Round", "Driver"]).reset_index(drop=True)

combined_fp2.to_csv("fp2_combined_2022_to_2025.csv", index=False)

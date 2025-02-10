# UNDER DEVELOPMENT
# This requires R to be installed

import subprocess, os, importlib.resources

def select(Preferences, method='EGA'):
    temp_data_path = 'data.csv'
    Preferences.preferences_df = Preferences.preferences_df[Preferences.preferences_df["same_choice_probs"] != -1]
    dat = Preferences.preferences_df[['prompt_id', 'name', 'same_choice_probs']] \
        .pivot(index='prompt_id', columns='name', values='same_choice_probs')
    dat.to_csv(temp_data_path, index=False)

    if (len(dat)/len(dat.columns) < 9 or len(dat.columns) < 20):
        print("Not enough data to run bootEGA. Running EGA instead.")
        method = 'EGA'

    if method == 'EGA':
        subprocess.run(["Rscript", importlib.resources.files("c3ai") / "data/Rscripts/EGA.R"])
    elif method == 'bootEGA':
        subprocess.run(["Rscript", importlib.resources.files("c3ai") / "data/Rscripts/bootEGA.R"])

    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
        print(f"Temp file {temp_data_path} removed")

    if os.path.exists('Rplots.pdf'):
        os.remove('Rplots.pdf')
        print(f"Rplots temp file removed")
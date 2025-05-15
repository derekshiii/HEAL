import pandas as pd
import os

def process_csv_files(file_list, output_file):
    all_data = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"File {f} does not exist. Skipping.")
            continue

        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except (pd.errors.EmptyDataError, KeyError) as e:
            print(f"Error processing file {f}: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Unknown error while processing file {f}: {e}. Skipping.")
            continue

    if not all_data:
        print("No files were successfully processed. Cannot generate summary.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed.")
        return

    summary_df = combined_df.loc[combined_df.groupby('FileName')['Edge_metrics'].idxmax()]
    summary_df = summary_df.sort_values(by='Edge_metrics', ascending=False)

    try:
        summary_df.to_excel(output_file, index=False)
        print(f"Summary saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    file_list = [
    ]
    output_file = "G:\\SampleSelection.xlsx"
    process_csv_files(file_list, output_file)

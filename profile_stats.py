import pandas as pd
import glob
import pandas
import numpy as np

def profile_data_to_df(path: str = './prof/*.prof'):
    # Get a list of all profiling result files

    files = glob.glob(path)  # Replace 'path_to_files' with the actual path to your profiling files
    # Initialize an empty list to store the data
    data = []

    # Read each file and extract relevant information
    total_durations = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Initialize variables to store file-specific information
        current_file = None
        file_duration = None
        last_line_num = None
        last_source_code = None
        triggered_by = None

        # Process each line
        for line in lines:
            line = line.strip()

            if line.startswith("Total duration"):
                total_duration = float(line.split(': ')[1].split('s')[0])
                total_durations.append(total_duration)

            elif line.startswith('File:'):
                # Extract the file information
                current_file = line.split(': ')[1].strip().split('site-packages')[-1]
            elif line.startswith('File duration:'):
                # Extract the file duration
                file_duration = float(line.split(': ')[1].split('s')[0])
            elif line.startswith('Line #'):
                # Skip the line containing column headers
                continue
            elif line.startswith('(call)'):
                # Split the (call) line to extract relevant information
                parts = line.split('|')
                if len(parts) >= 6:
                    line_num = last_line_num  # Use the last known line number
                    hits = int(parts[1].strip())
                    time = float(parts[2].strip())
                    time_per_hit = float(parts[3].strip())
                    percentage = float(parts[4].strip().replace('%', ''))
                    source_code = parts[5].strip().split('site-packages')[-1]  # Use the last known source code

                    # Append the extracted information to the data list
                    data.append({
                        'File': current_file,
                        'File duration': file_duration,
                        'Call': True,
                        'Line #': line_num,
                        'Hits': hits,
                        'Time': time,
                        'Time per hit': time_per_hit,
                        '%': percentage,
                        'Source code': source_code,
                        'Triggered by': last_source_code
                    })
            else:
                # Extract the line number and source code for future use
                parts = line.split('|')
                if len(parts) >= 6:
                    line_num = parts[0].strip()
                    source_code = parts[5].strip().split('site-packages')[-1]

                    # Append the extracted information to the data list
                    data.append({
                        'File': current_file,
                        'File duration': file_duration,
                        'Call': False,
                        'Line #': line_num,
                        'Hits': int(parts[1].strip()),
                        'Time': float(parts[2].strip()),
                        'Time per hit': float(parts[3].strip()),
                        '%': float(parts[4].strip().replace('%', '')),
                        'Source code': source_code,
                        'Triggered by': triggered_by
                    })

                    last_line_num = line_num
                    last_source_code = source_code

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    return total_durations, df


def aggregate_profile_data(df: pandas.DataFrame):
    # Calculate the mean, median, and standard deviation for each line in a file
    grouped_data = df.groupby(['File', 'Call', 'Line #', "Source code"])['Time'].agg(['mean', 'median', 'std']).reset_index()

    # Rename the columns
    grouped_data = grouped_data.rename(columns={'mean': 'mean_time', 'median': 'median_time', 'std': 'std_time'})

    # Round the columns to two decimal places
    grouped_data['mean_time'] = grouped_data['mean_time'].round(2)
    grouped_data['median_time'] = grouped_data['median_time'].round(2)
    grouped_data['std_time'] = grouped_data['std_time'].round(2)

    # Merge the values of 'Source code' and 'Triggered by' columns
    merged_df = grouped_data.merge(df.groupby(['File', 'Call', 'Line #', "Source code"])[['Triggered by']].first(), on=['File', 'Call', 'Line #'])

    # Convert 'Line #' column to numeric data type
    merged_df['Line #'] = pd.to_numeric(merged_df['Line #'])

    # Sort the DataFrame by 'File' and 'Line #' columns
    sorted_df = merged_df.sort_values(by=['File', 'Line #'])

    return sorted_df



if __name__ == "__main__":
    profile_data_path = './prof/*.prof'

    total_durations, df = profile_data_to_df(profile_data_path)
    sorted_df = aggregate_profile_data(df)
    sorted_df.to_csv('profile_stats.csv', index=False)
    
    min_time = 1  # Set the minimum mean time threshold
    filtered_df = sorted_df[(sorted_df['File'] != '<string>') & (sorted_df.groupby('File')['mean_time'].transform(lambda x: x >= min_time))]
    filtered_df.drop_duplicates(inplace=True)
    # filtered_df.to_csv('profile_stats_summary.txt', sep='\t', index=False)
    filtered_df = filtered_df.to_string(index=False)

    mean_duration = np.round(np.mean(total_durations), 2)
    median_duration = np.round(np.median(total_durations), 2)
    min_duration = np.round(np.min(total_durations), 2)
    max_duration = np.round(np.max(total_durations), 2)
    num_runs = np.round(len(total_durations), 2)

    stats = [
        f"Number of runs: {num_runs}",
        f"Mean duration: {mean_duration}",
        f"Median duration: {median_duration}",
        f"Min duration: {min_duration}",
        f"Max duration: {max_duration}",
    ]

    with open('profile_stats_summary.txt', 'w') as f:
        f.seek(0, 0)
        f.write('\n'.join(stats) + '\n' + '---'*25 + '\n' + filtered_df)

    print("Number of runs:", num_runs)
    print("Mean duration:", mean_duration)
    print("Median duration:", median_duration)
    print("Min duration:", min_duration)
    print("Max duration:", max_duration)
    print(filtered_df)
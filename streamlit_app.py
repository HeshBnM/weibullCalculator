import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize
import warnings
import base64

bin_sizes = {
    "4": [(320, 40), (50, 130), (140, 220), (230, 310), (361,999)],
    "8": [(338, 22), (22, 66), (68, 112), (112, 156), (158, 202), (202, 246), (248, 292), (292, 336), (361,999)],
    "12": [(345, 14), (15, 44), (45, 74), (75, 104), (105, 134), (135, 164), (165, 194), (195, 224), (225, 254), (255, 284), (285, 314), (315, 344), (361,999)],
    "16": [(349, 10), (11, 33), (34, 55), (56, 78), (79, 100), (101, 123), (124, 145), (146, 168), (169, 190), (191, 213), (214, 235), (236, 258), (259, 280), (281, 303), (304, 325), (326, 348), (361,999)],
    "20": [(351, 361), (9, 26), (27, 44), (45, 62), (63, 80), (81, 98), (99, 116), (117, 134), (135, 152), (153, 170), (171, 188), (189, 206), (207, 224), (225, 242), (243, 260), (261, 278), (279, 296), (297, 314), (315, 332), (333, 350), (361,999)],
    "24": [(352, 361), (8, 22), (22, 36), (38, 52), (52, 66), (68, 82), (82, 96), (98, 112), (112, 126), (128, 142), (142, 156), (158, 172), (172, 186), (188, 202), (202, 216), (218, 232), (232, 246), (248, 262), (262, 276), (278, 292), (292, 306), (308, 322), (322, 336), (338, 352), (361,999)],
    "28": [(354, 361), (6, 18), (19, 31), (32, 44), (45, 57), (58, 70), (71, 83), (84, 95), (96, 108), (109, 121), (122, 134), (135, 147), (148, 160), (161, 173), (174, 185), (186, 198), (199, 211), (212, 224), (225, 237), (238, 250), (251, 263), (264, 275), (276, 288), (289, 301), (302, 314), (315, 327), (328, 340), (341, 353), (361,999)],
    "32": [(354, 361), (6, 16), (17, 27), (28, 38), (39, 50), (51, 61), (62, 72), (73, 83), (84, 95), (96, 106), (107, 117), (118, 128), (129, 140), (141, 151), (152, 162), (163, 173), (174, 185), (186, 196), (197, 207), (208, 218), (219, 230), (231, 241), (242, 252), (253, 263), (264, 275), (276, 286), (287, 297), (298, 308), (309, 320), (321, 331), (332, 342), (343, 353), (361,999)],
    "36": [(355, 361), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 84), (85, 94), (95, 104), (105, 114), (115, 124), (125, 134), (135, 144), (145, 154), (155, 164), (165, 174), (175, 184), (185, 194), (195, 204), (205, 214), (215, 224), (225, 234), (235, 244), (245, 254), (255, 264), (265, 274), (275, 284), (285, 294), (295, 304), (305, 314), (315, 324), (325, 334), (335, 344), (345, 354), (361,999)]
}

seasons = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2]
}

def load_wind_data(file_path):
    # Load the wind data from the CSV file
    data = pd.read_csv(file_path, parse_dates=['ob_time'])
    data['ob_time'] = pd.to_datetime(data['ob_time'], format='%d/%m/%Y %H:%M')  # Ensure 'ob_time' is datetime with the correct format
    return data

def manipulated_data(data):
# Amend the wind direction data where both wind direction and wind speed are zero
    data.loc[(data['mean_wind_dir'] == 0) & (data['mean_wind_speed'] == 0), 'mean_wind_dir'] = 999

    # Change all 0 values in mean_wind_dir to 360
    data.loc[data['mean_wind_dir'] == 0, 'mean_wind_dir'] = 360

    return data

def extract_wind_speed(data):
    # Extract wind speed data
    return data['mean_wind_speed'].values

def calculate_probabilities(data, bins):
    # Extract the required columns
    extracted_data = data[['ob_time', 'mean_wind_dir', 'mean_wind_speed']]

    # Amend the wind direction data where both wind direction and wind speed are zero
    extracted_data.loc[(extracted_data['mean_wind_dir'] == 0) & (extracted_data['mean_wind_speed'] == 0), 'mean_wind_dir'] = 999

    # Change all 0 values in new_mean_wind_dir to 360
    extracted_data.loc[extracted_data['mean_wind_dir'] == 0, 'mean_wind_dir'] = 360

    # Adjust the bins to handle circularity
    adjusted_bins = []
    for bin_start, bin_end in bins:
        if bin_start <= bin_end:
            adjusted_bins.append((bin_start, bin_end))
        else:
            adjusted_bins.append((bin_start, 361))
            adjusted_bins.append((0, bin_end))

    # Define the bins 
    bin_counts = {bin_val: 0 for bin_val in adjusted_bins}
    for direction in extracted_data['mean_wind_dir']:
        # Adjust direction to fit into the range [0, 360)
        direction = direction % 1000
        for bin_val in adjusted_bins:
            if bin_val[0] <= direction <= bin_val[1]:
                bin_counts[bin_val] += 1
                break

    # Merge probabilities of bins ending at 361 and starting at 0
    merged_bincounts = {}
    skip_next = False  # Flag to skip the next bin if combined
    for i, (bin_val, prob) in enumerate(bin_counts.items()):
        if skip_next:
            skip_next = False
            continue
        
        if bin_val[1] == 361:  # If the current bin ends with 361
            # Check if the next bin starts with 0
            next_bin_val = list(bin_counts.keys())[i + 1]
            if next_bin_val[0] == 0:
                combined_prob = prob + bin_counts[next_bin_val]
                merged_bincounts[(bin_val[0], next_bin_val[1])] = combined_prob
                skip_next = True
            else:
                merged_bincounts[bin_val] = prob
        else:
            merged_bincounts[bin_val] = prob

    # If the last bin ends with 361, add it separately
    if list(bin_counts.keys())[-1][1] == 361 and not skip_next:
        last_bin_val = list(bin_counts.keys())[-1]
        merged_bincounts[last_bin_val] = bin_counts[last_bin_val]

    # Uniformly distribute samples of bin 999 across other bins
    total_samples = sum(merged_bincounts.values())
    num_bins = len(bins) - 1
    add_bin_999 = merged_bincounts[(361, 999)] // num_bins

    # Distribute count of bin 999 uniformly across other bins
    for bin_val in merged_bincounts:
        if bin_val != (361, 999):
            merged_bincounts[bin_val] += add_bin_999

    # Calculate probabilities
    probabilities = {bin_val: count / total_samples for bin_val, count in merged_bincounts.items()}

    return probabilities

def objective_function(k_test, wind_speed, bin999):
    # Define the objective function for optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k_numerator_term1 = np.sum(np.nan_to_num(wind_speed**k_test, nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(np.log(wind_speed), nan=0.0, posinf=0.0, neginf=0.0))
        k_denominator_term1 = np.sum(np.nan_to_num(wind_speed**k_test, nan=0.0, posinf=0.0, neginf=0.0))
        
        k_numerator_term2 = np.sum(np.nan_to_num(np.log(wind_speed), nan=0.0, posinf=0.0, neginf=0.0))
        k_denominator_term2 = len(wind_speed) + bin999

        k = 1 / ((k_numerator_term1/k_denominator_term1) - (k_numerator_term2/k_denominator_term2))
        return np.abs(k_test - k)

def perform_minimization(k_test, wind_speed, bin999):
    # Perform minimization
    result = minimize(objective_function, x0=k_test, args=(wind_speed, bin999), method='BFGS')
    return result.x[0]

def calculate_estimates(wind_speed, k_test_optimal, bin999):
    # Recalculate k and c with the optimal k_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k_numerator_term1 = np.sum(np.nan_to_num(wind_speed**k_test_optimal, nan=0.0, posinf=0.0, neginf=0.0) * np.nan_to_num(np.log(wind_speed), nan=0.0, posinf=0.0, neginf=0.0))
        k_denominator_term1 = np.sum(np.nan_to_num(wind_speed**k_test_optimal, nan=0.0, posinf=0.0, neginf=0.0))

        k_numerator_term2 = np.sum(np.nan_to_num(np.log(wind_speed), nan=0.0, posinf=0.0, neginf=0.0))
        k_denominator_term2 = len(wind_speed) + bin999

        # Calculate the maximum likelihood estimate of k.
        k = 1 / ((k_numerator_term1/k_denominator_term1) - (k_numerator_term2/k_denominator_term2))

        # Calculate the maximum likelihood estimate of c.
        c = ((1 / (len(wind_speed)+bin999) * np.sum(np.nan_to_num(wind_speed**k_test_optimal, nan=0.0, posinf=0.0, neginf=0.0))) ** (1 / k))

        return k, c

def filter_data_by_season(data, season_months):
    # Filter the data by the given months for a season
    return data[data['ob_time'].dt.month.isin(season_months)]
    
def main():
    st.sidebar.title('Weibull Calculator')
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_wind_data(uploaded_file)
        amended_data = manipulated_data(data)
        wind_speed = extract_wind_speed(amended_data)
        wind_speed_df = pd.DataFrame(wind_speed, columns=['mean_wind_speed'])

        selected_bin_size = st.sidebar.selectbox("Select bin size:", list(bin_sizes.keys()))

        spring_months = st.sidebar.multiselect("Spring months:", list(range(1, 13)), default=[3, 4, 5])
        summer_months = st.sidebar.multiselect("Summer months:", list(range(1, 13)), default=[6, 7, 8])
        autumn_months = st.sidebar.multiselect("Autumn months:", list(range(1, 13)), default=[9, 10, 11])
        winter_months = st.sidebar.multiselect("Winter months:", list(range(1, 13)), default=[12, 1, 2])

        seasons = {
            "Spring": spring_months,
            "Summer": summer_months,
            "Autumn": autumn_months,
            "Winter": winter_months
        }

        if st.sidebar.button("Submit"):
            bins = bin_sizes[selected_bin_size]
            all_months = [month for season in seasons.values() for month in season]
            extended_seasons = {**seasons, "Annual": all_months}

            combined_output_data = []

            for season, months in extended_seasons.items():
                st.write(f"\nCalculating for {season}...")

                seasonal_data = filter_data_by_season(amended_data, months)
                wind_speed_seasonal = extract_wind_speed(seasonal_data)
                wind_speed_df_seasonal = pd.DataFrame(wind_speed_seasonal, columns=['mean_wind_speed'], index=seasonal_data.index)

                probabilities = calculate_probabilities(seasonal_data, bins)

                bin999 = (probabilities[(361, 999)]) * len(wind_speed_df_seasonal) / float(selected_bin_size)
                del probabilities[(361, 999)]

                k_values = {}
                c_values = {}
                output_data = []

                for bin_val in probabilities.keys():
                    if bin_val[0] >= bin_val[1]:
                        mask1 = (seasonal_data['mean_wind_dir'] >= bin_val[0]) & (seasonal_data['mean_wind_dir'] < 361)
                        mask2 = (seasonal_data['mean_wind_dir'] >= 0) & (seasonal_data['mean_wind_dir'] < bin_val[1])

                        wind_speed_bin_1 = wind_speed_df_seasonal.loc[mask1[mask1.index.intersection(wind_speed_df_seasonal.index)]]
                        wind_speed_bin_2 = wind_speed_df_seasonal.loc[mask2[mask2.index.intersection(wind_speed_df_seasonal.index)]]

                        wind_speed_bin = pd.concat([wind_speed_bin_1, wind_speed_bin_2], ignore_index=True)
                    else:
                        mask = (seasonal_data['mean_wind_dir'] >= bin_val[0]) & (seasonal_data['mean_wind_dir'] < bin_val[1])
                        mask = mask.reindex(wind_speed_df_seasonal.index, fill_value=False)
                        wind_speed_bin = wind_speed_df_seasonal[mask]

                    k_test_optimal = perform_minimization(1, wind_speed_bin['mean_wind_speed'].values, bin999)

                    k_bin, c_bin = calculate_estimates(wind_speed_bin['mean_wind_speed'].values, k_test_optimal, bin999)

                    k_values[bin_val] = k_bin
                    c_values[bin_val] = c_bin

                    output_data.append([f"Bin {bin_val}", k_bin, c_bin, probabilities[bin_val]])

                k_test_optimal = perform_minimization(1, wind_speed_df_seasonal['mean_wind_speed'].values, bin999)

                k_all, c_all = calculate_estimates(wind_speed_df_seasonal['mean_wind_speed'].values, k_test_optimal, bin999)

                p_all = sum(probabilities.values())

                st.write(f"\nFor all wind speeds in {season}:")
                st.write(f"k = {k_all}, c = {c_all}, p = {p_all}")

                combined_output_data.extend([[season] + row for row in output_data])

                # Display output in a table
                st.write("\nOutput:")
                df_output = pd.DataFrame(output_data, columns=["Bin", "k", "c", "p"])
                st.write(df_output)

            # Download button for combined output
            combined_output_df = pd.DataFrame(combined_output_data, columns=["Season", "Bin", "k", "c", "p"])
            csv = combined_output_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            link = f'<a href="data:file/csv;base64,{b64}" download="combined_output.csv">Download combined output CSV file</a>'
            st.sidebar.markdown(link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
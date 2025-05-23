import pandas as pd
import re
import math
import statistics
from scipy.stats import t

input_file_name = "FinLoRA_Scores.csv"
output_file_name = "FinLoRA_Confidence_Intervals.csv"
confidence_level = 0.95

data_frame = pd.read_csv(input_file_name, index_col=0).drop(columns=["Llama 3.1 70B"])
list_pattern = re.compile(r"\[[^\]]+\]")

t_critical_values_by_sample_size = {}

def half_width_and_standard_deviation(cell_content):
    match_object = list_pattern.search(str(cell_content))
    if not match_object:
        return cell_content
    numeric_strings = [element.strip() for element in match_object.group(0).strip("[]").split(",")]
    numeric_values = []
    for element in numeric_strings:
        try:
            numeric_values.append(float(element))
        except ValueError:
            return cell_content
    sample_size = len(numeric_values)
    if sample_size < 2:
        return cell_content
    sample_standard_deviation = statistics.stdev(numeric_values)
    degrees_of_freedom = sample_size - 1
    probability_cutoff = 1 - (1 - confidence_level) / 2
    t_critical_value = t.ppf(probability_cutoff, degrees_of_freedom)
    t_critical_values_by_sample_size[sample_size] = t_critical_value
    half_width = t_critical_value * sample_standard_deviation / math.sqrt(sample_size)
    return f"[±{half_width:.3f}, {sample_standard_deviation:.3f}]"

result_frame = data_frame.applymap(half_width_and_standard_deviation)
result_frame.to_csv(output_file_name)

for sample_size, t_critical_value in sorted(t_critical_values_by_sample_size.items()):
    print(f"sample size = {sample_size}, degrees of freedom = {sample_size - 1}, t critical value ({int(confidence_level * 100)} percent confidence) = {t_critical_value:.3f}")

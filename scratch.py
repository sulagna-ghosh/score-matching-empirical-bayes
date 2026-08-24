from experiment_heteroscedastic import generate_and_save_synthetic_data, read_experiment_dict

print("###START####")
generate_and_save_synthetic_data([10, 20], 3, "c")

experiment_data_dict = read_experiment_dict("c")

print("############")
print(experiment_data_dict)
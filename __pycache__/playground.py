# import pandas as pd
# import re
# from datetime import datetime

# # Define patterns to extract relevant information
# param_pattern = re.compile(r"Starting training with hyperparameters: (.*)")
# epoch_pattern = re.compile(r"Epoch (\d+) - Training Loss: .*, Training Accuracy: (.*)")
# validation_pattern = re.compile(r"Epoch \d+ - Validation Loss: .*, Validation Accuracy: (.*)")
# test_pattern = re.compile(r"Epoch \d+ - Test Accuracy: (.*)")
# final_test_pattern = re.compile(r"Final test accuracy: (.*)")
# date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")  # Date in format YYYY-MM-DD

# # Data storage
# data = {
#     "d_state": [],
#     "d_conv": [],
#     "expand": [],
#     "batch_size": [],
#     "dropout_rate": [],
#     "num_mamba_layers": [],
#     "n_mfcc": [],
#     "n_fft": [],
#     "hop_length": [],
#     "n_mels": [],
#     "noise_level": [],
#     "lr": [],
#     "weight_decay": [],
#     "train_accuracy": [],
#     "validation_accuracy": [],
#     "test_accuracy": [],
#     "final_test_accuracy": [],
#     "epochs": [],
# }

# # Parse the log file from a specific start date and filter by final test accuracy
# def parse_log(file_path, start_date=None, min_final_test_accuracy=None):
#     with open(file_path, "r") as f:
#         current_params = None
#         final_test_accuracy = None
#         epoch_count = 0
#         train_accuracy = None
#         validation_accuracy = None
#         test_accuracy = None

#         for line in f:
#             # Check for date in the log
#             date_match = date_pattern.search(line)
#             if date_match:
#                 log_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
#                 if start_date and log_date < start_date:
#                     continue  # Skip lines before the start date

#             # Capture hyperparameters
#             param_match = param_pattern.search(line)
#             if param_match:
#                 if current_params and epoch_count > 0:
#                     # Only save data if all relevant fields are collected
#                     if final_test_accuracy is not None and (min_final_test_accuracy is None or final_test_accuracy >= min_final_test_accuracy):
#                         # Save previous model data
#                         data["epochs"].append(epoch_count)
#                         data["train_accuracy"].append(train_accuracy)
#                         data["validation_accuracy"].append(validation_accuracy)
#                         data["test_accuracy"].append(test_accuracy)
#                         data["final_test_accuracy"].append(final_test_accuracy)
                
#                 # Reset for new training
#                 current_params = eval(param_match.group(1))
#                 final_test_accuracy = None
#                 epoch_count = 0
#                 data["d_state"].append(current_params["d_state"])
#                 data["d_conv"].append(current_params["d_conv"])
#                 data["expand"].append(current_params["expand"])
#                 data["batch_size"].append(current_params["batch_size"])
#                 data["dropout_rate"].append(current_params["dropout_rate"])
#                 data["num_mamba_layers"].append(current_params["num_mamba_layers"])
#                 data["n_mfcc"].append(current_params["n_mfcc"])
#                 data["n_fft"].append(current_params["n_fft"])
#                 data["hop_length"].append(current_params["hop_length"])
#                 data["n_mels"].append(current_params["n_mels"])
#                 data["noise_level"].append(current_params["noise_level"])
#                 data["lr"].append(current_params["lr"])
#                 data["weight_decay"].append(current_params["weight_decay"])

#             # Capture epoch, training, and validation accuracies
#             epoch_match = epoch_pattern.search(line)
#             if epoch_match:
#                 epoch_count += 1
#                 train_accuracy = float(epoch_match.group(2))

#             validation_match = validation_pattern.search(line)
#             if validation_match:
#                 validation_accuracy = float(validation_match.group(1))

#             test_match = test_pattern.search(line)
#             if test_match:
#                 test_accuracy = float(test_match.group(1))

#             final_test_match = final_test_pattern.search(line)
#             if final_test_match:
#                 final_test_accuracy = float(final_test_match.group(1))

#         # Append the last model's data after the loop if it meets the accuracy threshold
#         if current_params and epoch_count > 0 and final_test_accuracy is not None and (min_final_test_accuracy is None or final_test_accuracy >= min_final_test_accuracy):
#             data["epochs"].append(epoch_count)
#             data["train_accuracy"].append(train_accuracy)
#             data["validation_accuracy"].append(validation_accuracy)
#             data["test_accuracy"].append(test_accuracy)
#             data["final_test_accuracy"].append(final_test_accuracy)

# # Ensure all lists in the data dictionary have the same length
# def check_data_lengths(data):
#     lengths = {k: len(v) for k, v in data.items()}
#     min_length = min(lengths.values())
    
#     for key in data:
#         if len(data[key]) > min_length:
#             data[key] = data[key][:min_length]

# # Call the function to parse the log
# # Provide a start date in 'YYYY-MM-DD' format and a minimum final test accuracy
# start_date = datetime.strptime("2024-09-17", "%Y-%m-%d")  # Example start date
# min_final_test_accuracy = 92  # Example minimum final test accuracy

# parse_log("optuna.log", start_date, min_final_test_accuracy)

# # Check and trim data to ensure consistent lengths
# check_data_lengths(data)

# # Create a pandas DataFrame
# df = pd.DataFrame(data)

# # Save to CSV
# df.to_csv("filtered_model_training_data.csv", index=False)

# print("Data has been saved to filtered_model_training_data.csv")



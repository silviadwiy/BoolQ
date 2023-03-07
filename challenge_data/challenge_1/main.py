import pandas as pd


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    
    ground_truth = pd.read_csv(test_annotation_file)
    prediction = pd.read_csv(user_submission_file)
    
    merged_df22 = pd.merge(ground_truth, prediction, on=["title", "answer"])
    correct_predictions = merged_df22.shape[0]
    
    total_predictions = ground_truth.shape[0]
    accuracy = correct_predictions / total_predictions
    print(kwargs["submission_metadata"])
    output = {}
    if phase_codename == "train":
        print("Evaluating for Train Phase")
        output["result"] = [
            {
                "train_split": {
                    "Score": accuracy,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Train Phase")
    elif phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "Score": accuracy,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Dev Phase")
    return output

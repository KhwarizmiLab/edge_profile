from data_engineering import all_data
from architecture_prediction import get_arch_pred_model

data = all_data(<Path_to_profile_folder>)
arch_pred_model = get_arch_pred_model("rf", df=data)  # "rf" = random forest, "lr" = linear regression, "nn" = neural net ...
print(arch_pred_model.evaluateTest())
print(arch_pred_model.evaluateTrain())
raw_data_path = "data/churn.txt"
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
col_name_types = {
    "State": "O",
    "Account Length": "int64",
    "Area Code": "O",
    "Phone": "O",
    "Int'l Plan": "O",
    "VMail Plan": "O",
    "VMail Message": "int64",
    "Day Mins": "float64",
    "Day Calls": "int64",
    "Day Charge": "float64",
    "Eve Mins": "float64",
    "Eve Calls": "int64",
    "Eve Charge": "float64",
    "Night Mins": "float64",
    "Night Calls": "int64",
    "Night Charge": "float64",
    "Intl Mins": "float64",
    "Intl Calls": "int64",
    "Intl Charge": "float64",
    "CustServ Calls": "int64",
    "Churn?": "O",
}

customer_id = "Phone"
rename_cols = {"Churn?": "is_churn"}

label_col = "is_churn"

import pandas as pd
from config import (
    raw_data_path,
    col_name_types,
    customer_id,
    rename_cols,
    train_data_path,
    test_data_path,
    label_col,
)


def split_train_test_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test set
    """

    train = df.sample(frac=1 - test_size, random_state=42)
    test = df.drop(train.index)

    return train, test


if __name__ == "__main__":

    print("Reading raw data...")
    df = pd.read_csv(raw_data_path)
    df = df.astype(col_name_types)

    print("Cleaning raw data...")

    df = df.drop_duplicates(subset=customer_id)
    df = df.rename(columns=rename_cols).set_index(customer_id)
    print("Total number of customer: {}".format(df.shape[0]))

    print("Convert categorical features to numeric variables...")
    data = pd.get_dummies(df)
    data = data.drop(columns=["is_churn_False."], axis=1).rename(
        columns={"is_churn_True.": label_col}
    )

    print("Total number of churned customer: {}".format(data["is_churn"].sum()))
    print("Total number of features: {}".format(data.shape[1] - 1))
    print("Creating train and test set...")
    train, test = split_train_test_data(data)
    train.to_csv(train_data_path)
    test.to_csv(test_data_path)
    print(
        f"Done! Dataset is saved as {train_data_path} and {test_data_path} from the current directory."
    )

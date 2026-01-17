import ingest

def main():
    # read
    csv_path = "sample_transactions.csv"
    print("dataset_id:", ingest.compute_dataset_id(csv_path))
    df = ingest.read_csv_bytes(csv_path)
    # print(df.head())

    # process
    df_clean, df_rejected, report = ingest.validate_and_clean(df)
    print(df_clean.head())
    print(df_rejected.head())
    print(report)



if __name__ == "__main__":
    main()
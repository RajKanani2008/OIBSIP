from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Retail Sales Data Analysis
# Oasis Infobyte Internship - Data Analytics
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "retail_sales.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


def main():
    # 1. Load data
    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "Transaction ID", "Date", "Customer ID", "Gender",
        "Age", "Product Category", "Quantity",
        "Price per Unit", "Total Amount"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    # 2. Basic data cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = [
        "Age", "Quantity", "Price per Unit", "Total Amount"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_rows = len(df)
    missing_before = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Exclude records with invalid dates from time-based analysis
    invalid_dates = int(df["Date"].isna().sum())
    analysis_df = df.dropna(subset=["Date"]).copy()

    # 3. Data quality report
    quality_report = pd.DataFrame({
        "Missing Values": missing_before,
        "Data Type": df.dtypes.astype(str)
    })

    quality_report.to_csv(
        REPORT_DIR / "data_quality_report.csv",
        index_label="Column"
    )

    # 4. Descriptive statistics
    descriptive_stats = analysis_df[
        numeric_columns
    ].describe().T

    descriptive_stats["mode"] = [
        analysis_df[column].mode().iloc[0]
        if not analysis_df[column].mode().empty else None
        for column in descriptive_stats.index
    ]

    descriptive_stats.to_csv(
        REPORT_DIR / "descriptive_statistics.csv"
    )

    # 5. Sales KPIs
    total_revenue = analysis_df["Total Amount"].sum()
    total_transactions = analysis_df["Transaction ID"].nunique()
    average_transaction = analysis_df["Total Amount"].mean()
    median_transaction = analysis_df["Total Amount"].median()
    unique_customers = analysis_df["Customer ID"].nunique()

    kpis = pd.DataFrame({
        "Metric": [
            "Total Revenue",
            "Unique Transactions",
            "Average Transaction Amount",
            "Median Transaction Amount",
            "Unique Customers",
            "Original Rows",
            "Rows After Duplicate Removal",
            "Invalid Dates Excluded"
        ],
        "Value": [
            total_revenue,
            total_transactions,
            round(average_transaction, 2),
            round(median_transaction, 2),
            unique_customers,
            original_rows,
            len(df),
            invalid_dates
        ]
    })

    kpis.to_csv(REPORT_DIR / "sales_kpis.csv", index=False)

    # 6. Monthly sales trend
    monthly_sales = (
        analysis_df
        .set_index("Date")
        .resample("MS")["Total Amount"]
        .sum()
    )

    monthly_sales.to_csv(
        REPORT_DIR / "monthly_sales.csv",
        header=["Total Revenue"]
    )

    plt.figure(figsize=(11, 6))
    monthly_sales.plot(marker="o", linewidth=2)
    plt.title("Monthly Retail Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_sales_trend.png", dpi=150)
    plt.close()

    # 6A. Quarterly sales summary
    quarterly_sales = (
        analysis_df
        .set_index("Date")
        .resample("QS")["Total Amount"]
        .sum()
    )

    quarterly_sales.to_csv(
        REPORT_DIR / "quarterly_sales.csv",
        header=["Total Revenue"]
    )
    # 7. Revenue by product category
    category_revenue = (
        analysis_df.groupby("Product Category")["Total Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    category_revenue.to_csv(
        REPORT_DIR / "category_revenue.csv",
        header=["Total Revenue"]
    )

    plt.figure(figsize=(9, 6))
    category_revenue.plot(kind="bar")
    plt.title("Revenue by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_revenue.png", dpi=150)
    plt.close()

    # 8. Customer demographics
    gender_counts = analysis_df["Gender"].value_counts()
    gender_counts.to_csv(
        REPORT_DIR / "gender_distribution.csv",
        header=["Transactions"]
    )

    plt.figure(figsize=(7, 5))
    gender_counts.plot(kind="bar")
    plt.title("Transactions by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=analysis_df,
        x="Age",
        bins=15,
        kde=True
    )
    plt.title("Customer Age Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_age_distribution.png", dpi=150)
    plt.close()

    # 9. Correlation heatmap
    correlation = analysis_df[
        numeric_columns
    ].corr(numeric_only=True)

    correlation.to_csv(REPORT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 10. Top products/categories and business recommendations
    category_summary = (
        analysis_df.groupby("Product Category")
        .agg(
            Transactions=("Transaction ID", "nunique"),
            Total_Revenue=("Total Amount", "sum"),
            Average_Transaction=("Total Amount", "mean")
        )
        .sort_values("Total_Revenue", ascending=False)
    )

    category_summary.to_csv(
        REPORT_DIR / "category_summary.csv"
    )

    top_category = category_summary.index[0]
    lowest_category = category_summary.index[-1]

    recommendations = [
        (
            f"Review {top_category} performance and test targeted "
            "promotions to maintain its revenue contribution."
        ),
        (
            f"Investigate {lowest_category} for customer demand, "
            "pricing, and product availability before changing strategy."
        ),
        (
            "Use monthly revenue trends to plan inventory and marketing "
            "around stronger and weaker sales periods."
        )
    ]

    with open(
        REPORT_DIR / "business_recommendations.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("BUSINESS RECOMMENDATIONS\n\n")
        for number, recommendation in enumerate(recommendations, start=1):
            file.write(f"{number}. {recommendation}\n\n")

    # 11. Console summary
    print("\n" + "=" * 55)
    print("RETAIL SALES DATA ANALYSIS")
    print("=" * 55)
    print(f"Original rows: {original_rows}")
    print(f"Rows after duplicate removal: {len(df)}")
    print(f"Missing values before cleaning: {int(missing_before.sum())}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Invalid dates excluded: {invalid_dates}")
    print(f"Date range: {analysis_df['Date'].min().date()} to "
          f"{analysis_df['Date'].max().date()}")
    print(f"Total revenue: {total_revenue:,.2f}")
    print(f"Unique transactions: {total_transactions}")
    print(f"Average transaction amount: {average_transaction:,.2f}")
    print(f"Median transaction amount: {median_transaction:,.2f}")
    print(f"Unique customers: {unique_customers}")

    print("\nRevenue by category:")
    print(category_revenue.to_string())

    print("\nBusiness recommendations:")
    for number, recommendation in enumerate(recommendations, start=1):
        print(f"{number}. {recommendation}")

    print(f"\nCharts saved to: {OUTPUT_DIR}")
    print(f"Reports saved to: {REPORT_DIR}")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()

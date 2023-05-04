# importing necessary libraries
import os
import sys
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("test_pyspark").getOrCreate()

# Problem 1: Raw Data Processing

# Adding these symbols while importing
Symbol_etf = "ETFS"
Symbol_stock = "STOCKS"

# Defining paths: making directory for storing the data
os.chdir("../")
parent_dir = os.getcwd()
directory = "\\Data"
path = parent_dir + directory
if not os.path.exists(f"{path}\\bronze\\etfs"):
    os.makedirs(f"{path}\\bronze\\etfs")
# #
if not os.path.exists(f"{path}\\bronze\\stocks"):
    os.makedirs(f"{path}\\bronze\\stocks")

# Defining Bronze Layer
path_bronze = path + "\\Bronze"

path_etf = path_bronze + "\\etfs"

path_stock = path_bronze + "\\stocks"

path_destination_etf = path_bronze + "\\etfs_parquet"

path_destination_stock = path_bronze + "\\stocks_parquet"

load_path_destination_etf = path_destination_etf + "\\*"

load_path_destination_stock = path_destination_stock + "\\*"

# Defining Silver Layer
transformed_path_etfs = path + "\\Silver\\Final_Averages_dataset_etf.parquet"

transformed_path_stock = path + "\\Silver\\Final_Averages_dataset_stocks.parquet"

# Defining Gold Layer
path_gold = path + "\\Gold"
if not os.path.exists(path_gold):
    os.makedirs(path_gold)

# If data is not in ETFS or STOCKS, the program will stop
if len(os.listdir(path_etf)) != 0:
    print("ETF Data found")
if len(os.listdir(path_stock)) != 0:
    print("Stock Data found")
else:
    print("empty directory")
    sys.exit()

# Defining Schema
schema = StructType(
    [
        StructField("Date", StringType(), True),
        StructField("Open", DoubleType(), True),
        StructField("High", DoubleType(), True),
        StructField("Low", DoubleType(), True),
        StructField("Close", DoubleType(), True),
        StructField("AdjClose", DoubleType(), True),
        StructField("Volume", DoubleType(), True),
        StructField("Symbol", StringType(), True),
        StructField("SecurityName", StringType(), True),
    ]
)


# Defining Class for Part1 and Part2
class ExtractTransformLoad:
    # Load files to memory, Files for ETFS and Stocks- Loading function, Loading first 40 files - otherwise kept getting out of memory error
    def load_files(self, loadpath, Symbol):
        """This function takes an empty dictionary and loads the file names as keys and dataframes as values. At the
        same time, this function is also adding a column Symbol - signifying the key without the .csv So, if a symbol
        is AAAY.csv, it will only take the company name as a column - AAAY For security name, we are adding it as a
        parameter of the function while calling the object of the class. The function returns the dictionary. Here as
        you can see, the variable n has been defined as I tried many optimization techniques but my system is having
        fewer cores and ram, I decided to take only first 40 data files.
        """
        n = 0
        dict_data = {}
        for i in os.listdir(loadpath):
            n += 1
            dict_data[i] = (
                spark.read.format("csv")
                .schema(schema)
                .option("header", "true")
                .load(f"{loadpath}/{i}")
                .withColumn("Symbol", lit(i[:-4]))
                .withColumn("SecurityName", lit(Symbol))
            )
            if n == 40:
                break
        return dict_data

    # Write files to disk in parquet format where key is file name and value is dataframe
    def writeparquet(self, writepath, dictionary):
        """This function helps in writing the loaded files into parquet format in the disk.
        Key represents the name of the file in the dictionary.
        Value represents the dataframe belonging to that particular key[:-4] will get rid of .csv at the end
        """
        for key, value in dictionary.items():
            value.coalesce(1).write.parquet(f"{writepath}/{key[:-4]}.parquet")

    # Part 2
    # Writing Files to Bronze after adding 2 columns
    def transformation(self, source_transform_path, destination_transform_path):
        """This function converts string format date column to timestamp format date column. After conversion to
        timestamp, using to_date function to convert to datetype column because directly, it wouldn't work. Making 2
        more columns out of date column like Month & Year as per requirements and later using them in window function to
        calculate moving average of volume and rolling median of adjusted close.
        Finally, we are returning the dataframe which contains the resultant columns with actual data
        """
        df_etfs_parquet = spark.read.load(source_transform_path)
        df_etfs_parquet = df_etfs_parquet.withColumn(
            "Date_timestamp", df_etfs_parquet.Date.cast("timestamp")
        )
        df_etfs_parquet = df_etfs_parquet.withColumn(
            "Date", to_date(col("Date_timestamp"))
        ).drop("Date_timestamp")
        df = df_etfs_parquet.withColumn("Month", F.month("Date")).withColumn(
            "year", F.year("Date")
        )
        w = Window.partitionBy("Symbol", "Month", "year").orderBy(
            "Symbol", "Date", "Month", "year"
        )
        df2 = df.withColumn("vol_moving_avg", F.round(avg("Volume").over(w), 1))
        df_final_averages = (
            df2.withColumn("id", F.lit(1))
            .withColumn(
                "adj_close_rolling_med",
                F.expr("percentile(AdjClose,0.5)").over(w.rowsBetween(-1, 0)),
            )
            .drop("id", "Month", "year")
        )
        df_final_averages.coalesce(1).write.mode("overwrite").parquet(
            destination_transform_path
        )
        return df_final_averages


# defining object of class
obj = ExtractTransformLoad()
# Saving ETFS dictionary load results
etfs_dict = obj.load_files(path_etf, Symbol_etf)
# Saving Stocks dictionary load results
stocks_dict = obj.load_files(path_stock, Symbol_stock)
# Writing to parquet in destination etf path
obj.writeparquet(path_destination_etf, etfs_dict)
# Writing to parquet in destination stock path
obj.writeparquet(path_destination_stock, stocks_dict)
# Transforming ETFS data
data_etfs = obj.transformation(load_path_destination_etf, transformed_path_etfs)
# Transforming Stock data
data_stocks = obj.transformation(
    load_path_destination_stock, transformed_path_stock
)
# Combining both data for machine learning model
data = data_etfs.union(data_stocks)
# Storing data in Gold Layer
data.coalesce(1).write.parquet(f"{path_gold}\\data.parquet")

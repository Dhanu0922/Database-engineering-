
# Databricks notebook source
# DBTITLE 1,importing libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import * 

# COMMAND ----------

# DBTITLE 1,creating dataframe
df = spark.read.format('csv') \
    .option('sep', ',') \
    .option('header', 'true') \
    .option('escape', '"') \
    .option('inferSchema', 'true') \
    .load('/FileStore/tables/googleplyastore.csv')

# COMMAND ----------

# DBTITLE 1,checking table entries
df.count()

# COMMAND ----------

# DBTITLE 1,checking table
df.show(1)

# COMMAND ----------

# DBTITLE 1,checking schema
df.printSchema()

# COMMAND ----------

# DBTITLE 1,Data cleaning
df=df.drop("size", "Content Rating","Last Updated","Android Ver")

# COMMAND ----------

# DBTITLE 1,checking if the fields are dropped or not
df.show(2)

# COMMAND ----------

# DBTITLE 1,dropping one more field
df=df.drop('Current Ver')

# COMMAND ----------

# DBTITLE 1,again checking cleaned data
df.show(2)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col
df=df.withColumn("Reviews", col("Reviews").cast(IntegerType()))\
.withColumn("Installs",regexp_replace(col("Installs"),"[^0-9]",""))\
    .withColumn("Installs",col("Installs").cast(IntegerType()))\
        .withColumn("Price",regexp_replace(col("Price"),"[$]",""))\
            .withColumn("Price",col("Price").cast(IntegerType()))

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.createOrReplaceTempView("apps")

# COMMAND ----------

# MAGIC %sql select * from apps

# COMMAND ----------

# DBTITLE 1,top reviews give to apps
# MAGIC %sql select App,sum(Reviews) from apps
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# DBTITLE 1,Top 10 installs
# MAGIC %sql select App,Type,sum(Installs) from apps
# MAGIC group by 1,2
# MAGIC order by 3 desc

# COMMAND ----------

# DBTITLE 1,category wise distribution
# MAGIC %sql select Category,sum(Installs) from apps
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# DBTITLE 1,top paid apps
# MAGIC %sql select App,sum(Price) from apps
# MAGIC where Type='Paid'
# MAGIC group by 1
# MAGIC order by 2 desc
# MAGIC

# COMMAND ----------



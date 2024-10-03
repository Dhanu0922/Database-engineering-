// Databricks notebook source
// DBTITLE 1,Load the dataframes

// File location and type
val file_location = "/FileStore/tables/Training_Data.csv"
val file_type = "csv"

// CSV options
val infer_schema = "true"
val first_row_is_header = "true"
val delimiter = ","

// The applied options are for CSV files. For other file types, these will be ignored.
val loanDF = spark.read.format(file_type)
.option("inferSchema", infer_schema)
.option("header", first_row_is_header)
.option("sep", delimiter)
.load(file_location)

display(loanDF)

// COMMAND ----------

// DBTITLE 1,Print Schema of dataframes

loanDF.printSchema()

// COMMAND ----------

// DBTITLE 1,statistics of data

display(loanDF.describe())

// COMMAND ----------

// DBTITLE 1,EDA/ Exploratory Data Analysis Creating Temporary View

loanDF.createOrReplaceTempView("LoanData")

// COMMAND ----------

// DBTITLE 1,histogram of income
// MAGIC %sql
// MAGIC
// MAGIC select Income from LoanDataNew

// COMMAND ----------

// DBTITLE 1,histogram of age
// MAGIC %sql
// MAGIC
// MAGIC select Age from LoanData

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select Experience from LoanData

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select CURRENT_JOB_YRS from LoanData

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select CURRENT_HOUSE_YRS from LoanData

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select RISK_FLAG from LoanData

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(HOUSE_OWNERSHIP), HOUSE_OWNERSHIP from LoanData group by HOUSE_OWNERSHIP

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(CAR_OWNERSHIP), CAR_OWNERSHIP, count(Risk_Flag), Risk_Flag from LoanData group by CAR_OWNERSHIP, Risk_Flag

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(Profession), Profession from LoanData group by Profession

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select AVG(income) as avg_income, profession from LoanData group by profession;

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select profession, avg(Age) as avg_age, avg(income) as avg_income, avg(experience) as avg_experience from LoanData group by profession 

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select avg(age) as avg_age, profession from LoanData group by profession

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select avg(experience) as avg_experience, profession from LoanData group by profession

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select * from loandata

// COMMAND ----------


var StringfeatureCol = Array("Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE")

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

display(indexed)


// COMMAND ----------


import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

val indexers = StringfeatureCol.map { 
  colName => new StringIndexer()
  .setInputCol(colName)
  .setHandleInvalid("skip")
  .setOutputCol(colName + "_indexed")
}

val pipeline = new Pipeline()
.setStages(indexers)

val PredictLoanDF = pipeline.fit(loanDF).transform(loanDF)

// COMMAND ----------


val splits = PredictLoanDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + "Testing Rows: " + test_rows)

// COMMAND ----------


import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("Id", "Income", "Age", "Experience", "Married/Single_indexed", "House_Ownership_indexed", "Car_Ownership_indexed", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS")).setOutputCol("features")
val training = assembler.transform(train).select($"features", $"Risk_Flag".alias("label"))
training.show()

// COMMAND ----------


import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

val model = dt.fit(training)

println("Model Trained!")

// COMMAND ----------


val testing = assembler.transform(test).select($"features", $"Risk_Flag".alias("trueLabel"))
testing.show()

// COMMAND ----------


val prediction = model.transform(testing)
val predicted = prediction.select("features", "prediction", "trueLabel")
display(predicted)

// COMMAND ----------


val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("trueLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)

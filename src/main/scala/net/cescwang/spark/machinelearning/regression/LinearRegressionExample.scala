package net.cescwang.spark.machinelearning.regression

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

object LinearRegressionExample {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().master("local[4]").appName("Linear Regression Example").getOrCreate()
    val data = session.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
    /*val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()*/
    val glr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    val model = glr.fit(data)

    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    //LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD deprecated now
  }
}

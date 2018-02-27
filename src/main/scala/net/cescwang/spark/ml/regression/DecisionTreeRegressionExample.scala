package net.cescwang.spark.ml.regression

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object DecisionTreeRegressionExample {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Decision Tree Regression").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_linear_regression_data.txt")
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = new DecisionTree(Strategy.defaultStrategy(Algo.Regression)).run(training)
    val predictionAndObservations = test.map{ case LabeledPoint(observation,features) =>
      val prediction = model.predict(features)
      (prediction, observation)
    }
    val metrics = new RegressionMetrics(predictionAndObservations)
    println("MAE: "+metrics.meanAbsoluteError)      //MAE: 10.448423605850225
    println("MSE: "+metrics.meanSquaredError)     //MSE: 179.636472460884
    println("RMSE: "+metrics.rootMeanSquaredError)      //RMSE: 13.402853146285084
  }
}

package net.cescwang.spark.machinelearning.classification

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object DecisionTreeExample {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Decision Tree Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = new DecisionTree(Strategy.defaultStrategy(Algo.Classification)).run(training)
    val predictionAndLabels = test.map{ case LabeledPoint(label,features) =>
      val prediction = model.predict(features)
      (prediction,label)
    }
    println("Accuracy = " + new MulticlassMetrics(predictionAndLabels).accuracy)  //Accuracy = 1.0
  }
}

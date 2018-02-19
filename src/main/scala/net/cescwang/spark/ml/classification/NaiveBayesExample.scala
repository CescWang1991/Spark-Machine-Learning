package net.cescwang.spark.ml.classification

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object NaiveBayesExample {

  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Naive Bayes Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = new NaiveBayes().run(training)
    val predictionAndLabels = test.map{ case LabeledPoint(label,features) =>
      val prediction = model.predict(features)
      (label,prediction)
    }
    println("Accuracy = " + new MulticlassMetrics(predictionAndLabels).accuracy)  //Accuracy = 0.9523809523809523
  }
}

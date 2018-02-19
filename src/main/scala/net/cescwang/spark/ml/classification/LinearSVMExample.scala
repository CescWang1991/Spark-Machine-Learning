package net.cescwang.spark.ml.classification

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object LinearSVMExample {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Linear SVM Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)
    model.clearThreshold()

    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("acROC: "+auROC)  //1.0
  }
}
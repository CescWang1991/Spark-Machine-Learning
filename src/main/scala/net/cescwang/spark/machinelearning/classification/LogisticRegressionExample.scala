package net.cescwang.spark.machinelearning.classification

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object LogisticRegressionExample {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Logistic Regression Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val modelLBFGS = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)
    val scoreAndLabels = test.map { case LabeledPoint(label, features) =>
      val score = modelLBFGS.predict(features)
      (score, label)
    }
    println("Accuracy = " + new MulticlassMetrics(scoreAndLabels).accuracy)  //Accuracy = 0.9523809523809523

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val precision = metrics.precisionByThreshold()
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    val roc = metrics.roc()
    roc.foreach { case (f, t) =>
      println(s"false positive rate: $f, true positive rate: $t")
    }

    val threshold = metrics.thresholds()
    threshold.foreach(t=>println(s"threshold: $t"))
  }
}

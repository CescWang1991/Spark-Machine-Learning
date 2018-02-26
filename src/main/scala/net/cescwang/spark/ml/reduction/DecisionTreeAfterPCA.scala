package net.cescwang.spark.ml.reduction

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object DecisionTreeAfterPCA {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Decision Tree Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val labels = data.map(_.label)
    val features = data.map(_.features)

    val pca = new PCA(20).fit(features)
    val projected = data.map { p =>
      p.copy(features = pca.transform(p.features))
    }
    println(data.first())
    println(projected.first())

    val splits = projected.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = new DecisionTree(Strategy.defaultStrategy(Algo.Classification)).run(training)
    val predictionAndLabel = test.map{ case LabeledPoint(label,features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    println("Accuracy = "+new MulticlassMetrics(predictionAndLabel).accuracy)
    //Accuracy = 0.9047619047619048(20) compared with Accuracy = 1.0 in DecisionTreeExample
  }
}

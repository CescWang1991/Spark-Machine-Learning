package net.cescwang.spark.ml.clustering

import org.apache.spark.mllib.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

object KMeansClusterForAnomalyDetection {

  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder()
      .master("local[4]")
      .appName("Decision Tree Example")
      .getOrCreate()
      .sparkContext
    val path = "data/mllib/AdvancedAnalytics/kddcup.data"
    val dataSet = sc.textFile(path)
    val labelsAndData = dataSet.map{ line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1, 3)
      val label = buffer.remove(buffer.length-1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label,vector)
    }

    val features = labelsAndData.values.cache
    val kmeans = new BisectingKMeans().setK(100)
    val model = kmeans.run(features)

    val clusterLabelCount = labelsAndData.map { case (label,datum) =>
      val cluster = model.predict(datum)
      (cluster,label)
    }.countByValue
    clusterLabelCount.toSeq.sorted.foreach {
      case ((cluster,label),count) =>
        println(f"$cluster%1s$label%18s$count%8s")
    }
  }
}

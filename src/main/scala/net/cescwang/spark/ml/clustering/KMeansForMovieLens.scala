package net.cescwang.spark.ml.clustering

import net.cescwang.spark.ml.DataTransfer
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.SparkSession

object KMeansForMovieLens {

  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder()
      .master("local[4]")
      .appName("Decision Tree Example")
      .getOrCreate()
      .sparkContext

    val movieVectors = DataTransfer.setMovieVectors(sc)

    //K-Means Train
    val numClusters = 5
    val numIterations = 10
    val movieClusterModel = new KMeans()
      .setK(numClusters)
      .setMaxIterations(numIterations)
      .run(movieVectors)

    //K-Means predict cluster
    val predicteCluster = movieClusterModel.predict(movieVectors)
    println(predicteCluster.take(10).mkString(","))
  }
}

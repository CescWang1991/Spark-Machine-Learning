package net.cescwang.spark.machinelearning.reader

import org.apache.spark.SparkContext

class ReadTrainData(sc:SparkContext) extends Serializable {
  val trainingData = sc.textFile("./src/main/Resources/StumbleUpon/train.csv")
  val head = trainingData.first()
  val training = trainingData
    .map(line=>line.split("\t"))
    .mapPartitionsWithIndex { (idx, iter) =>
      if (idx == 0) iter.drop(1) else iter }
  val testData = sc.textFile("./src/main/Resources/StumbleUpon/test.csv")
  val test = testData
    .map(line=>line.split("\t"))
    .mapPartitionsWithIndex { (idx, iter) =>
      if (idx == 0) iter.drop(1) else iter }
}

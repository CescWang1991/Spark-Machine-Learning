package net.cescwang.spark.ml.fpm

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object FPGrowthExample {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder()
      .master("local[4]")
      .appName("Decision Tree Example")
      .getOrCreate()
      .sparkContext
    val path = "data/mllib/sample_fpgrowth.txt"
    val data = sc.textFile(path)
    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))
    //transactions.foreach(array=>println(array.deep.mkString(",")))
    val fpg = new FPGrowth().setMinSupport(0.4).setNumPartitions(4)
    val model = fpg.run(transactions)

    model.freqItemsets.filter(fi=>fi.items.length>=3).foreach(println(_))

    val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent .mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }
}

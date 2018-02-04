package net.cescwang.spark.machinelearning.reader

import org.apache.spark.{SparkConf, SparkContext}

object main extends App {
  val config = new SparkConf().setMaster("local[4]").setAppName("SparkMachineLearning")
  val sc = new SparkContext(config)
  /*val rud = new ReadUserData(sc)
  println("Users: "+rud.num_users+" ; Genders: "+rud.num_gender+" ; Occupations: "+rud.num_occupations+" ; Zipcodes: "+rud.num_zipcodes)
  */
  val rmd = new ReadMovieData(sc)
}

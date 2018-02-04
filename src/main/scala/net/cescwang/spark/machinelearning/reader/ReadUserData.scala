package net.cescwang.spark.machinelearning.reader

import org.apache.spark.{SparkConf, SparkContext}

class ReadUserData(sc:SparkContext) {

  val user_data = sc.textFile("./src/main/Resources/ml-100k/u.user")
  val user_fields = user_data.map(line=>line.split("\\|"))
  val num_users = user_fields.map(fields => fields(0)).count()
  val num_gender = user_fields.map(fields => fields(2)).distinct().count()
  val num_occupations = user_fields.map(fields => fields(3)).distinct().count()
  val num_zipcodes = user_fields.map(fields => fields(4)).distinct().count()

}

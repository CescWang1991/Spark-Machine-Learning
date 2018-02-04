package net.cescwang.spark.machinelearning.reader

import org.apache.spark.SparkContext

class ReadMovieData(sc:SparkContext) {

  val movie_Data = sc.textFile("./src/main/Resources/ml-100k/u.item")
  val num_movies = movie_Data.count()
  val movie_fields = movie_Data.map(line=>line.split("\\|"))
  val movie_years = movie_fields.map(field => field(2)).map(x=>convert_year(x))
  val movie_ages = movie_years.map(yr => 1998-yr).countByValue()

  private def convert_year(x:String):Int = {
    x.substring(7).toInt
  }
}

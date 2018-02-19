package net.cescwang.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, Rating}

object DataTransfer {

  /**
    * @param sc Spark Context
    * @return RDD[(Index, (Title, Seq[Genres]))]
    */
  def setTitlesAndGenres(sc: SparkContext) = {
    val movies = sc.textFile("data/mllib/ml-100k/u.item")
    val genres = sc.textFile("data/mllib/ml-100k/u.genre")
    //Map(Int -> genre)
    val genreMap = genres.filter(!_.isEmpty).map(line =>
      line.split("\\|")).map(array=> (array(1).toInt,array(0))).collectAsMap.toMap
    //(Index,(Title,ArrayBuffer[genre]))
    //(1,(Toy Story (1995),ArrayBuffer(Animation, Children's, Comedy)))
    val titlesAndGenres = movies.map(line => line.split("\\|")).map{ array =>
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter{ case(g,_) =>
        g == "1"
      }.map{ case(_, idx) => genreMap(idx) }
      (array(0).toInt, (array(1), genresAssigned))
    }
    titlesAndGenres
  }

  /**
    *
    * @param sc Spark Context
    * @return Movie Vectors
    */
  def setMovieVectors(sc: SparkContext) = {
    val rawData = sc.textFile("data/mllib/ml-100k/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map{ case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 10, 0.1)
    val movieFactors = alsModel.productFeatures.map { case (id, factor) =>
      (id, Vectors.dense(factor)) }
    val movieVectors = movieFactors.map(_._2)
    movieVectors
  }
}

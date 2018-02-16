package net.cescwang.spark.machinelearning.regression

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}

object DecisionTreeForecastForest {

  def load(session:SparkSession, path: String): RDD[LabeledPoint] = {
    val sc = session.sparkContext
    val rawData = sc.textFile(path)
    val data = rawData.map{ line =>
      val values = line.split(',').map(_.toDouble)
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble   //wilderness corresponds to 4 binary features, indicates which index has 1
      val soil = values.slice(14,54).indexOf(1.0).toDouble  //soid corresponds to 40 binary features
      val features = Vectors.dense(values.slice(0,10) :+ wilderness :+ soil)
      val label = values.last - 1 //Decision Trees require label begins as 0
      LabeledPoint(label,features)
    }
    data
  }

  def split(data:RDD[LabeledPoint]):(RDD[LabeledPoint],RDD[LabeledPoint]) = {
    val splits = data.randomSplit(Array(0.8,0.2))
    val training = splits(0)
    val testing = splits(1)
    (training,testing)
  }

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionAndLabels = data.map{ lp =>
      val prediction = model.predict(lp.features)
      (prediction,lp.label)
    }
    new MulticlassMetrics(predictionAndLabels)
  }

  def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionAndLabels = data.map{ lp =>
      val prediction = model.predict(lp.features)
      (prediction,lp.label)
    }
    new MulticlassMetrics(predictionAndLabels)
  }

  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().master("local[4]").appName("Decision Tree Regression").getOrCreate()
    val path = "data/mllib/AdvancedAnalytics/covtype.data"
    val data = load(session,path)
    val (trainData,testData) = split(data)
    trainData.cache()
    testData.cache()

    val model = DecisionTree.trainClassifier(
      trainData, 7, Map[Int,Int](10->4, 11->40),
      "entropy", 30, 300)
    println("Decision Tree Accuracy: "+getMetrics(model,testData).accuracy)
    val forest = RandomForest.trainClassifier(
      trainData, 7, Map(10 -> 4, 11 -> 40), 20,
      "auto", "entropy", 30, 300)
    println("Random Forest Accuracy: "+getMetrics(forest,testData).accuracy)  //Random Forest Accuracy: 0.963336930270798
  }
}

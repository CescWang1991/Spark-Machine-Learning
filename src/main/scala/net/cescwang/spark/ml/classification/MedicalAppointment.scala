package net.cescwang.spark.ml.classification

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.sql.SparkSession

object MedicalAppointment extends App {

  val path = "data/KaggleV2-May-2016.csv"
  val session = SparkSession.builder()
    .master("local[*]")
    .appName("medical appointment")
    .getOrCreate()
  val sqlContext = session.sqlContext
  var df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("nullValue", "")
    .csv(path)

  val csv = session.sparkContext.textFile(path)
  val header = csv.first()
  val data = csv.filter(_ != header)
  val labeledData = data.map{line =>
    val parts = line.split(",")
    val label = if(parts(13).equals("No")) 0 else 1
    // Features in Gender change to integers(F-0, M-1)
    val gender = if(parts(2).equals("F")) 0 else 1
    // Breaking Date Features ScheduledDay into Date and Time Components
    val scheduledDate = parts(3).split("T")(0).split("-")
    val scheduledTime = parts(3).split("T")(1).split("Z")(0).split(":")
    // Breaking Date Features AppointmentDay into Date Components
    val appointmentDate = parts(4).split("T")(0).split("-")
    LabeledPoint(label, Vectors.dense(gender,    //Gender
      scheduledDate(0).toInt, scheduledDate(1).toInt, scheduledDate(2).toInt,   //Scheduled Date
      scheduledTime(0).toInt, scheduledTime(1).toInt, scheduledTime(2).toInt,   //Scheduled Time
      appointmentDate(0).toInt, appointmentDate(1).toInt, appointmentDate(2).toInt,   //Appointment Date
      parts(5).toInt,     //Age
      parts(7).toInt,     //Scholarship
      parts(8).toInt,     //Hipertension
      parts(9).toInt,     //Diabetes
      parts(10).toInt,    //Alcoholism
      parts(12).toInt    //SMS_received
    ))
  }.filter(lp => lp.features.toArray(10) >= 0.0).cache()    // Removing Observations with Negative Age Values
  val splits = labeledData.randomSplit(Array(0.8, 0.2), seed = 11L)
  val train = splits(0)
  val test = splits(1)

  // Train the model by applying decision tree
  val decisionTreeModel = new DecisionTree(Strategy.defaultStrategy(Algo.Classification)).run(train)
  val decisionTreePredictionAndLabels = test.map{ case LabeledPoint(label,features) =>
    val prediction = decisionTreeModel.predict(features)
    (label,prediction)
  }
  println("Accuracy of Decision Tree Classifier = " + new MulticlassMetrics(decisionTreePredictionAndLabels).accuracy)
  // Accuracy of Decision Tree Classifier = 0.7923231035424825

  // Train the model by applying naive bayes
  val naiveBayesModel = new NaiveBayes().run(train)
  val naiveBayesPredictionAndLabels = test.map{ case LabeledPoint(label,features) =>
    val prediction = naiveBayesModel.predict(features)
    (label,prediction)
  }
  println("Accuracy of Naive Bayes Classifier = " + new MulticlassMetrics(naiveBayesPredictionAndLabels).accuracy)
  // Accuracy of Naive Bayes Classifier = 0.6422001639194973

  // Train the model by applying logistic regression
  val logisticRegressionModel = new LogisticRegressionWithLBFGS()
    .setNumClasses(2)
    .run(train)
  val logisticRegressionPredictionAndLabels = test.map{ case LabeledPoint(label,features) =>
    val prediction = naiveBayesModel.predict(features)
    (label,prediction)
  }
  println("Accuracy of logistic Regression Classifier = " + new MulticlassMetrics(logisticRegressionPredictionAndLabels).accuracy)
  // Accuracy of logistic Regression Classifier = 0.6422001639194973

  // Train the model by applying support vector machine
  val supportVectorMachineModel = SVMWithSGD.train(train, 100)
  val supportVectorMachinePredictionAndLabels = test.map{ case LabeledPoint(label,features) =>
    val prediction = naiveBayesModel.predict(features)
    (label,prediction)
  }
  println("Accuracy of Support Vector Machine = " + new MulticlassMetrics(supportVectorMachinePredictionAndLabels).accuracy)
  // Accuracy of Support Vector Machine = 0.6422001639194973
}

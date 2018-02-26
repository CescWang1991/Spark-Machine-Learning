package net.cescwang.spark.ml.reduction

import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object DecisionTreeAfterSVD {
  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("Decision Tree Example").getOrCreate().sparkContext
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    val labels = data.map(_.label)
    val features = data.map(_.features)

    val mat: RowMatrix = new RowMatrix(features)
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(20)
    println(svd.s)
  }
}

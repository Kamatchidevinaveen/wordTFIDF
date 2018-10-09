package com.spark.project;

import java.util.Arrays;
import java.util.List;

//import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.feature.RegexTokenizer;
//import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
//import org.w3c.dom.Document;
//import org.w3c.dom.Element;

import org.apache.spark.ml.feature.NGram;

import org.apache.spark.ml.feature.StopWordsRemover;

import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;

//import com.agi.bigdata.utils.ReportGenerator;

//import com.agi.bigdataanalysis.Tokenizer;
import org.apache.spark.ml.feature.CountVectorizer;

public class wordTFIDF {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkSession spark = SparkSession.builder().master("local").appName("wordTFIDF").getOrCreate();
		
		List<Row> fileList = Arrays.asList(
				  RowFactory.create(0, "HiIheardaboutSpark"),
				 RowFactory.create(1, "IwishJavacouldusecaseclassesIuseJava"),
				  RowFactory.create(2, "Logisticregressionmodelsareneat")
				 );

				StructType schema = new StructType(new StructField[]{
				  new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
				  new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
				});
		Dataset<Row> sentenceDataFrame = spark.createDataFrame(fileList, schema);
		
		//Tokenizer
		//Tokenizer tokenizer1 = new Tokenizer().setInputCol("sentence").setOutputCol("words");
		RegexTokenizer tokenizer = new RegexTokenizer().setPattern("").setInputCol("sentence").setOutputCol("words");
		Dataset<Row> wordsData = tokenizer.transform(sentenceDataFrame);
		
		//stopWordRemover
		/*StopWordsRemover remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");
		Dataset<Row> filteredData = remover.transform(wordsData);*/
		
		
		//customfilter
		StopWordsRemover customFilterWords = new StopWordsRemover().setInputCol("words").setOutputCol("filteredAgain");
		String[] removeUnwantedWords = { "a", "e","i","o","u" };
		customFilterWords.setStopWords(removeUnwantedWords);
		Dataset<Row> filteredDataAgain = customFilterWords.transform(wordsData);
		
		
		//Ngram
		NGram ngramTransformer = new NGram().setN(3).setInputCol("filteredAgain").setOutputCol("ngrams");
		Dataset<Row> ngramDataFrame = ngramTransformer.transform(filteredDataAgain);
		
		
	    //CountVector(or)TF
		CountVectorizerModel cvModel = new CountVectorizer()
				  .setInputCol("ngrams")
				  .setOutputCol("rawFeature")
				  .fit(ngramDataFrame);
		Dataset<Row> result = cvModel.transform(ngramDataFrame);
		String[] vocabularyWords = cvModel.vocabulary();
		List<Row> tfResult = result.collectAsList();
		
		//IDFModel
		IDF idf = new IDF().setInputCol("rawFeature").setOutputCol("features");
		IDFModel idfModel = idf.fit(result);
        Dataset<Row> rescaledData = idfModel.transform(result);
        List<Row> featuresRow = rescaledData.select("features").collectAsList();
		rescaledData.cache();
        
		
		//Displaying result
        for (int i = 0; i < featuresRow.size(); i++) {
        	
        	Row row1 = featuresRow.get(i);
			SparseVector sv1 = (SparseVector) row1.get(row1.size() - 1);

			Row row2 = tfResult.get(i);
			SparseVector sv2 = (SparseVector) row2.get(row2.size() - 1);

			int[] originalIndices = sv1.indices();
			for (int index : originalIndices) {
				System.out.println(vocabularyWords[index]+" "+String.valueOf(index)+" "+String.valueOf(sv2.apply(index))+" "+String.valueOf(sv1.apply(index)));
			}
	   
		}
       ngramDataFrame.select("ngrams").show(false);
     
		
	}
}


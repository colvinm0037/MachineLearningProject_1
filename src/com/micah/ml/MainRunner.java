package com.micah.ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
 
public class MainRunner {
	
	public static String DECISION_TREE = "DecisionTree";
	public static String BOOSTING = "Boosting";
	public static String NEURAL_NETWORK = "NeuralNetwork";
	public static String SVM = "SVM";
	public static String KNN = "KNN";

	// Used some code and help from:
	// http://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
	// http://weka.wikispaces.com/Use+WEKA+in+your+Java+code
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
  
	public static void main(String[] args) throws Exception {
		
		System.out.println("Beginning Micah Colvin's Machine Learning Project #1");
		
		// Build the student data
		BufferedReader studentTestFile = readDataFile("data/students_test.arff");
		Instances studentTest = new Instances(studentTestFile);
		studentTest.setClassIndex(1);

		List<Instances> studentData = new ArrayList<Instances>();
		
		for (int i = 10; i <= 100; i += 10) {
			BufferedReader students = readDataFile("data/Students_" + i + ".arff");
			Instances trainingData = new Instances(students);
			trainingData.setClassIndex(1);
			studentData.add(trainingData);
		}	

		// Build the bicycle data
		BufferedReader bicycleTestFile = readDataFile("data/bikes_cleaned_testing.arff");
		Instances bikesTest = new Instances(bicycleTestFile);
		bikesTest.setClassIndex(13);

		List<Instances> bikesData = new ArrayList<Instances>();
		
		for (int i = 100; i <= 100; i += 10) {
			BufferedReader bikes = readDataFile("data/bikes_cleaned_" + i + ".arff");
			Instances trainingData = new Instances(bikes);
			trainingData.setClassIndex(13);
			bikesData.add(trainingData);
		}	
			
		// Train the students data set		
		System.out.println("Starting training on the students data");
		Learn(studentData, studentTest, "students", DECISION_TREE, Arrays.asList(0, 1, 2)); // Uses J48 Tree
		Learn(studentData, studentTest, "students", BOOSTING, Arrays.asList(0, 1));  // Uses J48 Tree with AdaM1 Boosting
		Learn(studentData, studentTest, "students", NEURAL_NETWORK, Arrays.asList(0, 1)); // Uses Multilayer Perceptron
	 	Learn(studentData, studentTest, "students", SVM, Arrays.asList(1, 3)); // Uses LibSVM with Linear and Polynomial Kernels
		Learn(studentData, studentTest, "students", KNN, Arrays.asList(1, 2, 3, 4, 5, 10, 20)); // Uses IBk with K values of 1, 2, 3, 4, 5, 10, 20
		
		// Train the bikes data set
		System.out.println("Starting training on the bikes data");
		Learn(bikesData, bikesTest, "bikes", DECISION_TREE, Arrays.asList(0, 1, 2)); // Uses J48 Tree		
		Learn(bikesData, bikesTest, "bikes", BOOSTING, Arrays.asList(0, 1, 2));  // Uses J48 Tree with AdaM1 Boosting
		//Learn(bikesData, bikesTest, "bikes", NEURAL_NETWORK); // Uses Multilayer Perceptron
		Learn(bikesData, bikesTest, "bikes", SVM, Arrays.asList(0, 1, 2, 3)); // Uses LibSVM with Linear and Polynomial Kernels
		Learn(bikesData, bikesTest, "bikes", KNN, Arrays.asList(1, 2, 3, 4, 5, 10, 20)); // Uses IBk with K values of 1, 2, 3, 4, 5, 10, 20
		
		System.out.println("Program finished.");
	}
	
	public static void Learn(List<Instances> dataList, Instances testData, String datasetName, String type) throws Exception {
		Learn(dataList, testData, datasetName, type, Arrays.asList(0));
	}
	
	public static void Learn(List<Instances> dataList, Instances testData, String datasetName, String type, List<Integer> versions) throws Exception {
		
		System.out.println("\nStarting " + type);
		String fileName = type + "_" + datasetName + ".csv";					
		
		PrintWriter writer = new PrintWriter(fileName, "UTF-8");
		
		List<List<String>> trainingSets = new ArrayList<List<String>>();
		List<List<String>> testingSets = new ArrayList<List<String>>();
		List<List<String>> cvSets = new ArrayList<List<String>>();
		
		for (Integer i : versions) {

			List<String> trainingRates = new ArrayList<String>();
			List<String> testingRates = new ArrayList<String>();
			List<String> cvRates = new ArrayList<String>();
			
			for (Instances instances : dataList) {
				
				// Build the classifier with this instance and train it with the training data
				Classifier cls = buildClassifier(type, i);
				Evaluation cvEval = new Evaluation(instances);
				cvEval.crossValidateModel(cls, instances, 10, new Random(1));
				cvRates.add(cvEval.errorRate() * 100 + "%");				
				
				//Classifier cls = buildClassifier(type, i);
				cls.buildClassifier(instances);
				Evaluation eval = new Evaluation(instances);
				
				// Evaluate the classifier with the training data
				eval.evaluateModel(cls, instances);							
				trainingRates.add(eval.errorRate() * 100 + "%");
				
				// Evaluate the classifier with the testing data
				Evaluation testEval = new Evaluation(instances);
				testEval.evaluateModel(cls, testData);							
				testingRates.add(testEval.errorRate() * 100 + "%");
				//System.out.println(testEval.toSummaryString());
			}
			trainingSets.add(trainingRates);
			testingSets.add(testingRates);
			cvSets.add(cvRates);
		}
		
		for (int i = 0; i < trainingSets.size(); i++) {
			
			List<String> trainingRates = trainingSets.get(i);
			List<String> testingRates = testingSets.get(i);
			List<String> cvRates = cvSets.get(i);

			writer.println("Using Option: " + i);
			writer.println("Data Used, Training Error, Test Error, CV Error");
			for (int j = 0; j < trainingRates.size(); j++) {
				writer.println(((j + 1) * 10) + "," + trainingRates.get(j) + "," + testingRates.get(j) + "," + cvRates.get(j));
			}
			writer.println();
		}
		
		writer.close();
		System.out.println("Results saved to " + fileName);
	}
	
	// Build the classifier for the type passed in. Optionally also pass in an option parameter.
	public static Classifier buildClassifier(String type, int i) throws Exception {
		
		Classifier cls = null;
				
		if (type.equals(DECISION_TREE)) {

			J48 j48 = new J48();
			j48.setNumFolds(5);
			
			if (i == 1) {				
				String options = ( "-U");
				String[] optionsArray = options.split( " " );			
				cls = j48;
				((AbstractClassifier) cls).setOptions( optionsArray );	
				return cls;
			} else if (i == 2) {
				j48.setConfidenceFactor(.5f);				
			}
			
			cls = j48;			
			
		} else if (type.equals(BOOSTING)) {
						
			AdaBoostM1 ada = new AdaBoostM1();
			J48 j48 = new J48();			
			if (i == 1) {
				j48.setUnpruned(true);
			} else if (i == 2) {
				j48.setConfidenceFactor(.5f);
			}

			cls = ada;
			((AdaBoostM1)cls).setClassifier(j48);
			
		} else if (type.equals(NEURAL_NETWORK)) {
			
			MultilayerPerceptron p = new MultilayerPerceptron();			
			if (i == 1) {
				p.setDecay(true);
			}
			
			cls = p;	
			
			
		} else if (type.equals(SVM)) {
			
			LibSVM svm = new LibSVM();
			
			String options = ( "-K " + i + " -S 0");
			String[] optionsArray = options.split( " " );			
			cls = svm;
			((AbstractClassifier) cls).setOptions( optionsArray );
			
		} else if (type.equals(KNN)) {
			
			IBk ibk = new IBk();
			ibk.setKNN(i);		
			cls = ibk;
		}
		
		return cls;
	}
		
}
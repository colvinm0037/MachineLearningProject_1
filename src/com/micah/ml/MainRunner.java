package com.micah.ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
		
		// Borrowed some code from: http://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
		
		// TODO: Add Cross Validation!
		
		System.out.println("Beginning Micah Colvin's Machine Learning Project #1");
		
		// Build the student data
		BufferedReader testfile = readDataFile("data/students_test.arff");
		Instances test = new Instances(testfile);
		test.setClassIndex(1);

		List<Instances> studentData = new ArrayList<Instances>();
		
		for (int i = 10; i <= 100; i += 10) {
			BufferedReader students = readDataFile("data/Students_" + i + ".arff");
			Instances trainingData = new Instances(students);
			trainingData.setClassIndex(1);
			studentData.add(trainingData);
		}	
		
		// TODO: Build the other data set
		// For the Bike data
		// MP5 should work
		// Ibk works
		// Multilayer Perceptron works
		// Boosting?
		// SVM?
				
		// Call each of the five algorithms		
		Learn(studentData, test, DECISION_TREE); // Uses J48 Tree
		Learn(studentData, test, BOOSTING);  // Uses J48 Tree with AdaM1 Boosting
		Learn(studentData, test, NEURAL_NETWORK); // Uses Multilayer Perceptron
		Learn(studentData, test, SVM, Arrays.asList(0, 1)); // Uses LibSVM with Linear and Polynomial Kernels
		Learn(studentData, test, KNN, Arrays.asList(1, 2, 3, 4, 5, 10, 20)); // Uses IBk with K values of 1, 2, 3, 4, 5, 10, 20
		
		System.out.println("Program finished.");
	}
	
	public static void Learn(List<Instances> dataList, Instances testData, String type) throws Exception {
		Learn(dataList, testData, type, Arrays.asList(0));
	}
	
	public static void Learn(List<Instances> dataList, Instances testData, String type, List<Integer> versions) throws Exception {
		
		System.out.println("\nStarting " + type);
		String fileName = type + ".csv";					
		
		PrintWriter writer = new PrintWriter(fileName, "UTF-8");
		
		List<List<String>> trainingSets = new ArrayList<List<String>>();
		List<List<String>> testingSets = new ArrayList<List<String>>();
		
		for (Integer i : versions) {

			List<String> trainingRates = new ArrayList<String>();
		
			for (Instances instances : dataList) {
				
				// Build the classifier with this instance
				Classifier cls = buildClassifier(type, i);
				cls.buildClassifier(instances);
				
				// Evaluate the classifier with the test data
				Evaluation eval = new Evaluation(instances);
				eval.evaluateModel(cls, instances);			
				
				trainingRates.add(eval.errorRate() * 100 + "%");
			}
			trainingSets.add(trainingRates);
		}
		
		for (Integer i : versions) {
			
			List<String> testingRates = new ArrayList<String>();
									
			for (Instances instances : dataList) {
			
				// Build the classifier with this instance
				Classifier cls = buildClassifier(type, i);
				cls.buildClassifier(instances);
				
				// Evaluate the classifier with the test data
				Evaluation eval = new Evaluation(instances);
				
				eval.evaluateModel(cls, testData);
				
				// Print the result
				testingRates.add(eval.errorRate() * 100 + "%");		
			}
			testingSets.add(testingRates);			
		}
		
		for (int i = 0; i < trainingSets.size(); i++) {
			
			List<String> trainingRates = trainingSets.get(i);
			List<String> testingRates = testingSets.get(i);

			writer.println("Using Option: " + i);
			writer.println("Data Used, Training Error, Test Error");
			for (int j = 0; j < 10; j++) {
				writer.println(((j + 1) * 10) + "," + trainingRates.get(j) + "," + testingRates.get(j));
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
			cls = j48;
		} else if (type.equals(BOOSTING)) {
			cls = new AdaBoostM1();
			((AdaBoostM1)cls).setClassifier(new J48());			
		} else if (type.equals(NEURAL_NETWORK)) {
			MultilayerPerceptron p = new MultilayerPerceptron();			
			cls = p;
		} else if (type.equals(SVM)) {
			LibSVM svm = new LibSVM();
			String options = ( "-S " + i );
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
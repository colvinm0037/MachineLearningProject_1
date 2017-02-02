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
import weka.core.WekaPackageManager;
import weka.core.converters.LibSVMLoader;
import weka.core.converters.LibSVMSaver;
 
public class MainRunner {
	
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
		
		// Call each of the five algorithms		
		DecisionTree(studentData, test);
		Boosting(studentData, test);
		SVM(studentData, test);
		KNN(studentData, test);
		NeuralNetwork(studentData, test);
	
		System.out.println("Program finished.");
	}
	
	public static void DecisionTree(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("\nStarting Decision Tree");
							
		List<String> trainingRates = new ArrayList<String>();
		List<String> testingRates = new ArrayList<String>();
		PrintWriter writer = new PrintWriter("decision_tree.csv", "UTF-8");
		
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			Classifier j48 = new J48();			
			Classifier cls = j48;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			trainingRates.add(eval.errorRate() * 100 + "%");
		}
						
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			Classifier j48 = new J48();		
			Classifier cls = j48;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			testingRates.add(eval.errorRate() * 100 + "%");		
		}
		
		writer.println("Data Used, Training Error, Test Error");
		for (int i = 0; i < 10; i++) {
			writer.println(((i + 1) * 10) + "," + trainingRates.get(i) + "," + testingRates.get(i));
		}
		writer.close();
		System.out.println("Decision tree results saved to decision_tree.csv");
	}
	
	public static void Boosting(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("\nStarting Boosting Tree");
		
		List<String> trainingRates = new ArrayList<String>();
		List<String> testingRates = new ArrayList<String>();
		PrintWriter writer = new PrintWriter("boosting.csv", "UTF-8");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			Classifier cls = new AdaBoostM1();
			((AdaBoostM1)cls).setClassifier(new J48());
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			trainingRates.add(eval.errorRate() * 100 + "%");	
		}
						
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			Classifier cls = new AdaBoostM1();
			((AdaBoostM1)cls).setClassifier(new J48());			
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			testingRates.add(eval.errorRate() * 100 + "%");		
		}		
		
		writer.println("Data Used, Training Error, Test Error");
		for (int i = 0; i < 10; i++) {
			writer.println(((i + 1) * 10) + "," + trainingRates.get(i) + "," + testingRates.get(i));
		}
		writer.close();
		System.out.println("Boosting results saved to boosting.csv");
	}		
	
	public static void NeuralNetwork(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("\nStarting Neural Network (This usually takes a minute or two to finish)");
		
		List<String> trainingRates = new ArrayList<String>();
		List<String> testingRates = new ArrayList<String>();
		PrintWriter writer = new PrintWriter("neural_networks.csv", "UTF-8");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			MultilayerPerceptron p = new MultilayerPerceptron();			
			Classifier cls = p;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			trainingRates.add(eval.errorRate() * 100 + "%");
		}
		
		System.out.println("Test Results");
				
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			MultilayerPerceptron p = new MultilayerPerceptron();			
			Classifier cls = p;	
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			testingRates.add(eval.errorRate() * 100 + "%");
		}
		
		writer.println("Data Used, Training Error, Test Error");
		for (int i = 0; i < 10; i++) {
			writer.println(((i + 1) * 10) + "," + trainingRates.get(i) + "," + testingRates.get(i));
		}
		writer.close();
		System.out.println("Neural Network results saved to neural_networks.csv");
	}
	
	public static void SVM(List<Instances> dataList, Instances testData) throws Exception {
		
		// Some code borrow from: https://stackoverflow.com/questions/5223982/how-to-use-libsvm-with-weka-in-my-java-code
		
		System.out.println("\nStarting SVM");
		
		PrintWriter writer = new PrintWriter("svm.csv", "UTF-8");
		
		List<List<String>> trainingSets = new ArrayList<List<String>>();
		List<List<String>> testingSets = new ArrayList<List<String>>();
		
		List<Integer> kernels = Arrays.asList(0, 1);
		String[] kernelNames = new String[]{"Linear", "Polynomial", "Radial Basis Function", "Sigmoid"};
		
		for (Integer i : kernels) {

			List<String> trainingRates = new ArrayList<String>();
						
			for (Instances instances : dataList) {
				
				// Build the classifier with this instance
				LibSVM svm = new LibSVM();
				String options = ( "-S " + i );
				String[] optionsArray = options.split( " " );
				
				Classifier cls = svm;
				((AbstractClassifier) cls).setOptions( optionsArray );
				cls.buildClassifier(instances);
				
				// Evaluate the classifier with the test data
				Evaluation eval = new Evaluation(instances);
				eval.evaluateModel(cls, instances);			
				
				// Print the result
				trainingRates.add(eval.errorRate() * 100 + "%");
			}
			trainingSets.add(trainingRates);
		}
		
		for (Integer i : kernels) {
			
			List<String> testingRates = new ArrayList<String>();
			
			for (Instances instances : dataList) {
			
				// Build the classifier with this instance
				LibSVM svm = new LibSVM();
				String options = ( "-S " + i );
				String[] optionsArray = options.split( " " );
				
				Classifier cls = svm;
				((AbstractClassifier) cls).setOptions( optionsArray );
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

			writer.println("Using Kernel: " + kernelNames[i]);
			writer.println("Data Used, Training Error, Test Error");
			for (int j = 0; j < 10; j++) {
				writer.println(((j + 1) * 10) + "," + trainingRates.get(j) + "," + testingRates.get(j));
			}
			writer.println();
		}
		
		writer.close();
		System.out.println("SVM results saved to svm.csv");
	}
	
	public static void KNN(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("\nStarting KNN");
		PrintWriter writer = new PrintWriter("knn.csv", "UTF-8");
		
		List<List<String>> trainingSets = new ArrayList<List<String>>();
		List<List<String>> testingSets = new ArrayList<List<String>>();
		List<Integer> knn = Arrays.asList(1, 2, 3, 4, 5, 10, 20);
		
		for (Integer i : knn) {
			
			List<String> trainingRates = new ArrayList<String>();
			
			for (Instances instances : dataList) {
				
				// Build the classifier with this instance
				IBk ibk = new IBk();
				ibk.setKNN(i);
				
				Classifier cls = ibk;
				cls.buildClassifier(instances);
				
				// Evaluate the classifier with the test data
				Evaluation eval = new Evaluation(instances);
				eval.evaluateModel(cls, instances);			
				
				// Print the result
				trainingRates.add(eval.errorRate() * 100 + "%");
			}
			trainingSets.add(trainingRates);
		}
		
		for (Integer i : knn) {
		
			List<String> testingRates = new ArrayList<String>();	
			
			for (Instances instances : dataList) {
			
				// Build the classifier with this instance
				IBk ibk = new IBk();
				ibk.setKNN(i);
			
				Classifier cls = ibk;
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

			writer.println("Using KNN: " + i);
			writer.println("Data Used, Training Error, Test Error");
			for (int j = 0; j < 10; j++) {
				writer.println(((j + 1) * 10) + "," + trainingRates.get(j) + "," + testingRates.get(j));
			}
			writer.println();
		}
		
		writer.close();
		System.out.println("KNN results saved to knn.csv");
	}
	
}
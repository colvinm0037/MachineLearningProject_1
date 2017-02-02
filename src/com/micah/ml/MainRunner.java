package com.micah.ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
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
		//DecisionTree(studentData, test);
		//NeuralNetwork(studentData, test);
		//Boosting(studentData, test);
		SVM(studentData, test);
		//KNN(studentData, test);
		
	}
	
	public static void DecisionTree(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("STARTING Decision Tree\n");
		
		System.out.println("Training Results");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			Classifier j48 = new J48();			
			Classifier cls = j48;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");
		}
		
		System.out.println("Test Results");
				
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			Classifier j48 = new J48();		
			Classifier cls = j48;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");		
		}
		
	}
	
	public static void Boosting(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("STARTING Boosting Tree\n");
		
		System.out.println("Training Results");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			Classifier cls = new AdaBoostM1();
			((AdaBoostM1)cls).setClassifier(new J48());
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");
		}
		
		System.out.println("Test Results");
				
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			Classifier cls = new AdaBoostM1();
			((AdaBoostM1)cls).setClassifier(new J48());			
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");		
		}		
	}		
	
	public static void NeuralNetwork(List<Instances> dataList, Instances testData) throws Exception {
		
		System.out.println("STARTING Neural Network\n");
		
		System.out.println("Training Results");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			MultilayerPerceptron p = new MultilayerPerceptron();			
			Classifier cls = p;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");
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
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");		
		}
	}
	
	public static void SVM(List<Instances> dataList, Instances testData) throws Exception {
		
		// TODO: Use at least two Kernel Functions

		// Some code borrow from: https://stackoverflow.com/questions/5223982/how-to-use-libsvm-with-weka-in-my-java-code
		
		System.out.println("STARTING SVM\n");
		
		System.out.println("Training Results");
					
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			
			LibSVM svm = new LibSVM();
			Classifier cls = svm;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			eval.evaluateModel(cls, instances);			
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");
		}
		
		System.out.println("Test Results");
		
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			LibSVM svm = new LibSVM();
			Classifier cls = svm;
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, testData);
			
			// Print the result
			System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");		
		}
				
	}
	
	public static void KNN(List<Instances> dataList, Instances testData) throws Exception {
		System.out.println("STARTING KNN\n");
		
		List<Integer> knn = Arrays.asList(1, 2, 3, 4, 5, 10, 20);
		
		System.out.println("Training Results");
		for (Integer i : knn) {

			System.out.println("Using KNN: " + i);
			
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
				System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");
			}
			System.out.println();
		}
		
		System.out.println("Test Results");
		
		for (Integer i : knn) {
		
			System.out.println("Using KNN: " + i);
				
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
				System.out.println("Error Rate: " + eval.errorRate() * 100 + "%");		
			}
			System.out.println();
		}
	}
	
	
}
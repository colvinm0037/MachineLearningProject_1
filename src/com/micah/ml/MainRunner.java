package com.micah.ml;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
 
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
		
		BufferedReader testfile = readDataFile("data/students_test.arff");
		Instances test = new Instances(testfile);
		test.setClassIndex(1);

		List<Instances> dataList = new ArrayList<Instances>();
		
		for (int i = 10; i <= 100; i += 10) {
			BufferedReader students = readDataFile("data/Students_" + i + ".arff");
			Instances trainingData = new Instances(students);
			trainingData.setClassIndex(1);
			dataList.add(trainingData);
		}	
		
		System.out.println("Training");
		 
		for (Instances instances : dataList) {
		
			// Build the classifier with this instance
			Classifier cls = new J48();
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, instances);
			
			// Print the result
			System.out.println(eval.toSummaryString("\nTraining Results\n======\n", false));		
			
		}
		
		System.out.println("Test");
		
		for (Instances instances : dataList) {
			
			// Build the classifier with this instance
			Classifier cls = new J48();
			cls.buildClassifier(instances);
			
			// Evaluate the classifier with the test data
			Evaluation eval = new Evaluation(instances);
			
			eval.evaluateModel(cls, test);
			
			// Print the result
			System.out.println(eval.toSummaryString("\nTest Results\n======\n", false));
		}
	
 
	}
}
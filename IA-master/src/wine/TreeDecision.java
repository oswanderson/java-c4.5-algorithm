package wine;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class TreeDecision {
    public static void main(String[] args) {
        Instances training = null;
        Instances testing = null;
        
        try {
            BufferedReader readerTraining = new BufferedReader(new FileReader("/home/wanderson/Documents/UAM/ArtificialIntelligence/wineTraining.arff"));
            BufferedReader readerTesting = new BufferedReader(new FileReader("/home/wanderson/Documents/UAM/ArtificialIntelligence/wineTesting.arff"));
            training = new Instances(readerTraining);
            testing = new Instances(readerTesting);
            readerTraining.close();
            readerTesting.close();
        } catch(IOException e) {
            System.out.println(e.getMessage());
        }

        // setting class attribute
        // training.setClassIndex(training.numAttributes() - 1);
        training.setClassIndex(training.numAttributes() - 1);
        testing.setClassIndex(testing.numAttributes() - 1);

        //Configura o algoritmo a ser ensinado
        J48 model = new J48();
        model.setUnpruned(true);
        
        try {
            model.buildClassifier(training);
            
            Evaluation eval = new Evaluation(training);
            eval.evaluateModel(model, testing);
            
            System.out.println(
                    eval.toSummaryString("\nResult\n====================\n", true));
        } catch(Exception e) {
            System.out.println(e.getMessage());
        }
        
//        for (int i = 0; i < data.numInstances(); i++) {
//            double pred = j48.classifyInstance(data.instance(i));
//            System.out.print("ID: " + data.instance(i).value(0));
//            System.out.print(", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
//            System.out.println(", predicted: " + data.classAttribute().value((int) pred));
//        }
    }
}

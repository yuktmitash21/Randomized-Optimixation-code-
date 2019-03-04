package opt.test.admissionsTesting;

import dist.*;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class admissionsTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, outputLayer = 1, trainingIterations = 1000;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    //private static FeedForwardNetwork networks[] = new FeedForwardNetwork[100];
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[3];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[100];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[100];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    //private static String[] oaNames = new String[100];
    //private static String[] oaNames = new String[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";
    private static List<List<Double>> resultsToAverage = new ArrayList<>();

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, 6, 6, 6, 6, 6, 6, 6, 6,6, 6, 6, 6, 6, 6, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

         /*int m = 0;
        for (double i = 0.01; i <= 1; i += 0.01, m++) {
            oa[m] = new SimulatedAnnealing(1E11, i, nnop[m]);
            oaNames[m] = String.valueOf(i);
        }*/

        for (int i = 0; i < trainingIterations; i++) {
            resultsToAverage.add(new ArrayList<>());
        }

        //oa[0] = new RandomizedHillClimbing(nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for (int k = 0; k < 10; k++) {
            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 6, 6, 6, 6, 6, 6, 6, 6,6, 6, 6, 6, 6, 6, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
            }

            //oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
            //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

            for (int i = 1; i < 2; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i]); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances.length; j++) {
                    networks[i].setInputValues(instances[j].getData());
                    networks[i].run();

                    predicted = Double.parseDouble(instances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            }
        }

        System.out.println("Linear separator");

        for (int i = 0; i < resultsToAverage.size(); i++) {
            double sum = 0;
            for (int j = 0; j < resultsToAverage.get(i).size(); j++) {
                sum += resultsToAverage.get(i).get(j);
            }

            System.out.println(sum / (double) resultsToAverage.get(i).size());
        }

        //System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double lastError = 0;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
                lastError = error;
            }

            System.out.println(df.format(error));
            resultsToAverage.get(i).add(error);
        }

        //System.out.println(df.format(Double.parseDouble(oaName)) + " " + lastError);
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[500][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/admissionsTesting/admissionsMod.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[3]; // 4 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 3; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
package opt.test.admissionsTesting;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class testSAAt1E11 {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, outputLayer = 1, trainingIterations = 1000;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    //private static FeedForwardNetwork networks[] = new FeedForwardNetwork[100];
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[5];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[100];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[5];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[100];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[5];
    //private static String[] oaNames = new String[100];
    //private static String[] oaNames = new String[3];
    private static String[] oaNames = {"SA", "SA", "SA", "SA", "SA"};
    private static String results = "";
    private static List<List<Double>> oaResultsTrain = new ArrayList<>();
    private static List<List<Double>> oaResultsTest = new ArrayList<>();

    private static ArrayList<Double> train20 = new ArrayList<>();
    private static ArrayList<Double> train40 = new ArrayList<>();
    private static ArrayList<Double> train60 = new ArrayList<>();
    private static ArrayList<Double> train80 = new ArrayList<>();
    private static ArrayList<Double> train100 = new ArrayList<>();


    private static ArrayList<Double> test20 = new ArrayList<>();
    private static ArrayList<Double> test40 = new ArrayList<>();
    private static ArrayList<Double> test60 = new ArrayList<>();
    private static ArrayList<Double> test80 = new ArrayList<>();
    private static ArrayList<Double> test100 = new ArrayList<>();


    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        /*for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }*/

        /*int m = 0;
        for (double i = 0.01; i <= 1; i += 0.01, m++) {
            oa[m] = new SimulatedAnnealing(1E11, i, nnop[m]);
            oaNames[m] = String.valueOf(i);
        }*/

        for (int i = 0; i < trainingIterations; i++) {
            oaResultsTrain.add(new ArrayList<>());
            oaResultsTest.add(new ArrayList<>());
        }

        //oa[0] = new RandomizedHillClimbing(nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for (int k = 0; k < 3; k++) {
            new RandomOrderFilter().filter(set);
            TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
            ttsf.filter(set);
            DataSet train = ttsf.getTrainingSet();
            DataSet test = ttsf.getTestingSet();

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,6, 6, 6, 6, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], measure);
            }

            //oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[0] = new SimulatedAnnealing(1E2, .2, nnop[0]);
            oa[1] = new SimulatedAnnealing(1E2, .4, nnop[1]);
            oa[2] = new SimulatedAnnealing(1E2,.6, nnop[2]);
            oa[3] = new SimulatedAnnealing(1E2, .8, nnop[3]);
            oa[4] = new SimulatedAnnealing(1E2, .9, nnop[4]);

            //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

            for (int i = 0; i < 5; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i], train, test, i); //trainer.train();
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

        System.out.println("\nLinear separator\n");

        for (int i = 0; i < oaResultsTrain.size(); i++) {
            double trainSum = 0;
            double testSum = 0;

            for (int j = 0; j < oaResultsTrain.get(i).size(); j++) {
                trainSum += oaResultsTrain.get(i).get(j);
            }

            for (int j = 0; j < oaResultsTest.get(i).size(); j++) {
                testSum += oaResultsTest.get(i).get(j);
            }

            double first = trainSum / (double) oaResultsTrain.get(i).size();
            double second = testSum / (double) oaResultsTest.get(i).size();
            System.out.println(df.format(first) + " " + df.format(second));
        }

        //System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName, DataSet train, DataSet test, int coolLevel) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        Instance[] trainInstances = train.getInstances();
        Instance[] testInstances = test.getInstances();

        //double lastError = 0;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double trainError = 0;
            double testError = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                trainError += measure.value(output, example);
                //lastError = error;
            }

            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();

                Instance output = testInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                testError += measure.value(output, example);
                //lastError = error;
            }



            String cooling = "";
            if (coolLevel == 0) {
                cooling = "20";
                train20.add(trainError);
                test20.add(testError);
            } else if (coolLevel == 1) {
                cooling = "40";
                train40.add(trainError);
                test40.add(testError);

            } else if (coolLevel == 2) {
                cooling = "60";
                train60.add(trainError);
                test60.add(testError);

            } else if (coolLevel == 3) {
                cooling = "80";
                train80.add(trainError);
                test80.add(testError);

            } else if (coolLevel == 4) {
                cooling = "90";
                train100.add(trainError);
                test100.add(testError);

            }

            System.out.println("Cooling level: " + cooling + "Iteration " + String.format("%04d" ,i) + ": " + df.format(trainError) + " " + df.format(testError));
            oaResultsTrain.get(i).add(trainError);
            oaResultsTest.get(i).add(testError);
        }
        /*
        *  private static void makeCSV(List<Double> yList1, List<Double> yList2,
                                List<Double> yList3, List<Double> yList4,
                                List<Double> yList5,
                                String name, String yVal, String yval2, String yVal3, String
                                yVal4, String yVal5)*/
        makeCSV(train20, train40, train60, train80, train100, "src/opt/test/admissionsTesting/admissionsTrainSA1E11.csv",
                "20% cooling", "40% cooling", "60% cooling", "80% cooling", "90% cooling");

        makeCSV(test20, test40, test60, test80, test100, "src/opt/test/admissionsTesting/admissionsTestSA1E11.csv",
                "20% cooling", "40% cooling", "60% cooling", "80% cooling", "90% cooling");

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

    private static void makeCSV(List<Double> yList1, List<Double> yList2,
                                List<Double> yList3, List<Double> yList4,
                                List<Double> yList5,
                                String name, String yVal, String yval2, String yVal3, String
                                yVal4, String yVal5) {
        String temp = name;
        File newFile = new File(temp);
        try {
            FileWriter fw = new FileWriter(temp, true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);
            pw.println("Iterations," + yVal + "," + yval2 + "," + yVal3 + "," + yVal4 + "," + yVal5);
            for (int i = 1; i < 1001; i++) {
                pw.println(i + "," + yList1.get(i - 1) + "," + yList2.get(i - 1)
                + "," + yList3.get(i - 1) + "," + yList4.get(i - 1) + "," + yList5.get(i - 1));
            }
            pw.flush();
            pw.close();
            File dump = new File(temp);
            newFile.renameTo(dump);

        } catch (Exception e) {
            e.printStackTrace();
        }



    }
}

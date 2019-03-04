package opt.test.admissionsTesting;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
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
public class PopulationTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 4, outputLayer = 1, trainingIterations = 100;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static ArrayList<Double> pop1 = new ArrayList<>();
    private static ArrayList<Double> pop2 = new ArrayList<>();
    private static ArrayList<Double> pop3 = new ArrayList<>();
    private static ArrayList<Double> pop4 = new ArrayList<>();
    private static ArrayList<Double> pop5 = new ArrayList<>();


    private static ArrayList<Double> pop1test = new ArrayList<>();
    private static ArrayList<Double> pop2test = new ArrayList<>();
    private static ArrayList<Double> pop3test = new ArrayList<>();
    private static ArrayList<Double> pop4test = new ArrayList<>();
    private static ArrayList<Double> pop5test = new ArrayList<>();

    //private static FeedForwardNetwork networks[] = new FeedForwardNetwork[100];
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[5];
    //private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[100];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[5];

    //private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[100];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[5];
    //private static String[] oaNames = new String[100];
    //private static String[] oaNames = new String[3];
    private static String[] oaNames = {"GA","GA", "GA", "GA", "GA"};
    private static String results = "";
    private static List<List<Double>> oaResultsTrain = new ArrayList<>();
    private static List<List<Double>> oaResultsTest = new ArrayList<>();

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
        //oa[2] = new StandardGeneticAlgorithm(50, 100, 10, nnop[2]);

        for (int k = 0; k < 5; k++) {
            new RandomOrderFilter().filter(set);
            TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
            ttsf.filter(set);
            DataSet train = ttsf.getTrainingSet();
            DataSet test = ttsf.getTestingSet();

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], measure);
            }

            //oa[0] = new RandomizedHillClimbing(nnop[0]);
            //oa[1] = new SimulatedAnnealing(1E11, .35, nnop[1]);
            oa[0] = new StandardGeneticAlgorithm(200, 10, 10, nnop[0]);
            oa[1] = new StandardGeneticAlgorithm(200, 20, 10, nnop[1]);
            oa[2] = new StandardGeneticAlgorithm(200, 50, 10, nnop[2]);
            oa[3] = new StandardGeneticAlgorithm(200, 100, 10, nnop[3]);
            oa[4] = new StandardGeneticAlgorithm(200, 200, 10, nnop[4]);

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
            System.out.println(df.format(first / (double) 5021) + " " + df.format(second / (double) 2152));
        }

        //System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName, DataSet train, DataSet test, int ab) {
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

            System.out.println("Iteration " + String.format("%04d" ,i) + ": " + df.format(trainError / (double) 5021) + " " + df.format(testError / (double) 2152));
            oaResultsTrain.get(i).add(trainError);
            oaResultsTest.get(i).add(testError);

            if (ab == 0) {
                pop1.add(trainError / (double) 5021);
                pop1test.add(testError / (double) 2152);

            } else if (ab == 1) {
                pop2.add(trainError / (double) 5021);
                pop2test.add(testError / (double) 2152);

            } else if (ab == 2) {
                pop3.add(trainError / (double) 5021);
                pop3test.add(testError / (double) 2152);

            } else if (ab == 3) {
                pop4.add(trainError / (double) 5021);
                pop4test.add(testError / (double) 2152);

            } else if (ab == 4) {
                pop5.add(trainError / (double) 5021);
                pop5test.add(testError / (double) 2152);

            }

        }

        makeCSV(pop1, pop2, pop3, pop4, pop5, "src/opt/test/admissionsTesting/admissionsPopulationVaryTraining.csv",
                "Ten", "Twenty","Fifty", "One-Hundred", "One-Hundred-Fifty");
        makeCSV(pop1test, pop2test, pop3test, pop4test, pop5test, "src/opt/test/admissionsTesting/admissionsPopulationVaryTest.csv",
                "Ten", "Twenty","Fifty", "One-Hundred", "One-Hundred-Fifty");
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
                pw.println(i + "," + yList1.get(i) + "," + yList2.get(i)
                        + "," + yList3.get(i) + "," + yList4.get(i) + "," + yList5.get(i));
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
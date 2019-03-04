package opt.test.admissionsTesting;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class admissionsNNIterated {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    //private static int[] iterations = {1, 2, 3, 5, 10, 20, 25, 50, 100, 200, 400, 500, 750, 1000};
    private static int[] iterations = getArray();
    private static List<List<String>> oaResults = new ArrayList<>();
    private static String results = "";

    private static ArrayList<Double> testingErrorsRHC = new ArrayList<>();
    private static ArrayList<Double> trainingTimeRHC = new ArrayList<>();

    private static ArrayList<Double> testingErrorSA = new ArrayList<>();
    private static ArrayList<Double> trainingTimeSA = new ArrayList<>();

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oaResults.add(new ArrayList<>());
        oaResults.add(new ArrayList<>());
        oaResults.add(new ArrayList<>());

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length - 1; i++) {
            for (Integer k : iterations) {
                switch(i) {
                    case 0: oa[0] = new RandomizedHillClimbing(nnop[0]); break;
                    case 1: oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]); break;
                    case 2: oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]); break;
                }

                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i], k); //trainer.train();
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

                oaResults.get(i).add(df.format(correct / (correct + incorrect) * 100));

                results = k + ": ";
                results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                //{"RHC", "SA", "GA"};
                if (oaNames[i].equals("RHC")) {
                    testingErrorsRHC.add((incorrect / (correct + incorrect)));
                    trainingTimeRHC.add(trainingTime);

                } else if (oaNames[i].equals("SA")) {
                    testingErrorSA.add((incorrect / (correct + incorrect)));
                    trainingTimeSA.add(trainingTime);

                } else {

                }


                System.out.println(results);
            }
        }

        makeCSV(iterations, testingErrorsRHC, "src/opt/test/admissionsTesting/dataRHCTestingError.csv", "Testing Error");
        makeCSV(iterations, trainingTimeRHC, "src/opt/test/admissionsTesting/dataRHCTrainingTime.csv", "Training Time");

        makeCSV(iterations, testingErrorSA, "src/opt/test/admissionsTesting/dataSATestingError.csv", "Testing Error");
        makeCSV(iterations, trainingTimeSA, "src/opt/test/admissionsTesting/dataRHCTrainingError.csv", "Training Time");



        List<String> output_lines = new ArrayList<>();
        for (int i = 0, k = 0; i < oaResults.get(0).size(); i++, k += 25) {
            String s = k + "," + oaResults.get(0).get(i) + "," + oaResults.get(1).get(i) + "," + oaResults.get(2).get(i);
            output_lines.add(s);
        }

        try {
            Path file = Paths.get("src/opt/test/admissions_Iterated_Results.csv");
            Files.write(file, output_lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        modifyAdmissions();

        double[][][] attributes = new double[500][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/admissionsTesting/admissionsMod.csv")));

            for(int i = 0; i < attributes.length; i++) {
              // S tring x = br.readLine()
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

    public static void modifyAdmissions() {
        String temp = "src/opt/test/admissionsTesting/admissionsMod.csv";
        File oldFile = new File("src/opt/test/admissionsTesting/admissionsNoWords.csv");
        File newFile = new File(temp);

        String useless = "";
        String keep1 = "";
        String keep2 = "";
        String keep3 = "";
        String keep4 = "";
        String keep5 = "";
        String keep6 = "";
        String keep7 = "";
        String keep8 = "";

        try {
            FileWriter fw = new FileWriter(temp, true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);
            Scanner x = new Scanner(new File("src/opt/test/admissionsTesting/admissionsNoWords.csv"));
            x.useDelimiter("[,\n]");

            while (x.hasNext()) {
                useless = x.next();
                keep1 = x.next();
                keep2 = x.next();
                keep3 = x.next();
                keep4 = x.next();
                keep5 = x.next();
                keep6 = x.next();
                keep7 = x.next();
                keep8 = x.next();

                String classifier = Double.parseDouble(keep8) >= 0.725 ? 1 + "" : 0 + "";
                pw.println(keep6 + "," + keep1 + "," + keep3 + "," + classifier);
            }
            x.close();
            pw.flush();
            pw.close();
            File dump = new File(temp);
            newFile.renameTo(dump);
        } catch (Exception e) {
            e.printStackTrace();
        }



    }

    private static void makeCSV(int[] iterations, ArrayList<Double> testingErrorsRHC, String name, String yVal) {
        String temp = name;
        File newFile = new File(temp);
        try {
            FileWriter fw = new FileWriter(temp, true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);
            pw.println("Iterations," + yVal);
            for (int i: iterations) {
                pw.println(i + "," + testingErrorsRHC.get(i - 1));
            }
            pw.flush();
            pw.close();
            File dump = new File(temp);
            newFile.renameTo(dump);

        } catch (Exception e) {
            e.printStackTrace();
        }



    }

    public static int[] getArray() {
        int[] arr = new int[1000];
        for (int i = 0; i < 1000; i++) {
            arr[i] = i + 1;
        }

        return arr;
    }
}

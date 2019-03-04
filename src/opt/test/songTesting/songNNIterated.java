package opt.test.songTesting;

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

public class songNNIterated {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 7, hiddenLayer = 5, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static int[] iterations = {1, 2, 5, 10, 20, 25, 50, 100, 200, 400, 500, 750, 1000};
    private static List<List<String>> oaResults = new ArrayList<>();
    private static String results = "";

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

        for(int i = 0; i < oa.length; i++) {
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


                System.out.println(results);
            }
        }

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
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/songTesting/songMod.csv")));

            for(int i = 0; i < attributes.length; i++) {
                // S tring x = br.readLine()
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[14]; // 4 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 14; j++)
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
        String temp = "src/opt/test/songTesting/songMod.csv";
        File oldFile = new File("src/opt/test/songTesting/songDataNoWords.csv");
        File newFile = new File(temp);


        String keep1 = "";
        String keep2 = "";
        String keep3 = "";
        String keep4 = "";
        String keep5 = "";
        String keep6 = "";
        String keep7 = "";
        String keep8 = "";
        String keep9 = "";
        String keep10 = "";
        String keep11 = "";
        String keep12 = "";
        String keep13 = "";
        String keep14 = "";
        String keep15 = "";
        String keep16 = "";
        String keep17 = "";

        try {
            FileWriter fw = new FileWriter(temp, true);
            BufferedWriter bw = new BufferedWriter(fw);
            PrintWriter pw = new PrintWriter(bw);
            Scanner x = new Scanner(new File("src/opt/test/songTesting/songDataNoWords.csv"));
            x.useDelimiter(",");

            while (x.hasNext()) {
                keep1 = x.next();

                keep2 = x.next();
                keep3 = x.next();
                keep4 = x.next();
                keep5 = x.next();
                keep6 = x.next();
                keep7 = x.next();
                keep8 = x.next();
                keep9 = x.next();
                keep10 = x.next();
                keep11 = x.next();
                keep12 = x.next();
                keep13 = x.next();
                keep14 = x.next();
                keep15 = x.next();
                keep16 = x.next();
                keep17 = x.next();

                System.out.println(keep2 + "," + keep3 + "," + keep4 + "," + keep5 +
                        "," + keep6 + "," + keep7 + "," + keep8 + "," + keep9
                        + "," + keep10 + "," + keep11 + "," + keep12 + "," + keep13
                        + "," + keep14);

                pw.println(keep2 + "," + keep3 + "," + keep4 + "," + keep5 +
                        "," + keep6 + "," + keep7 + "," + keep8 + "," + keep9
                + "," + keep10 + "," + keep11 + "," + keep12 + "," + keep13
                + "," + keep14);
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
}

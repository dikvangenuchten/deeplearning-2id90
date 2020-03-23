package experiments;

import nl.tue.s2id90.dl.experiment.BatchResult;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class hyperParameterFinder {
    public static void main(String[] args) throws IOException {
        int[] batchSizes = {16, 32, 64, 128};
        double[] learningRates = {0.1, 0.05, 0.01};
        List<BatchResult> batchResults = new ArrayList<>();
        for (double learningRate : learningRates) {
            for (int batchSize : batchSizes) {
                ExperimentSCT exp = new ExperimentSCT();
                BatchResult result = exp.go(batchSize, learningRate);
                batchResults.add(result);
                System.out.println("RESULT: " + result.validation + " batchSize: " + batchSize + " lr: " + learningRate);
            }
        }
        for (BatchResult batchResult : batchResults) {
            System.out.println(batchResult.batch_id);
            System.out.println(batchResult.validation);
            System.out.println(batchResult.loss);
            System.out.println(batchResult.learning_rate);
            System.out.println(" ");
        }
    }

}

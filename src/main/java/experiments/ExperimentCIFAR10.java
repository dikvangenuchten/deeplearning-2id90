package experiments;

import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.LRELU;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.*;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.ADADELTA;
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
import nl.tue.s2id90.dl.NN.optimizer.update.L2Decay;
import nl.tue.s2id90.dl.NN.optimizer.update.MomentumGradientDescent;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.BatchResult;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.Cifar10Reader;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

import java.io.IOException;
import java.util.List;

public class ExperimentCIFAR10 extends GUIExperiment {
    // (hyper) parameters
    int batchSize = 32;
    int epochs = 10;
    double learningRate = 0.1;
    BatchResult localResult;
    ShowCase showCase;

    public BatchResult go(int batchSize, double learningRate) throws IOException {
        // you are going to add code here
        this.batchSize = batchSize;
        this.learningRate = learningRate;

        // read input and print some information on the data
        InputReader reader = new Cifar10Reader(batchSize, 5);

        System.out.println(" Reader info :\n" + reader.toString());

        List<String> label = Cifar10Reader.getLabelsAsString();

        showCase = new ShowCase(i -> label.get(i));

        FXGUI.getSingleton().addTab("Show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));

        TensorShape inputs = reader.getInputShape();
        TensorShape outputs = reader.getOutputShape();

        Model model = createModel(inputs, outputs);

        model.initialize(new Gaussian());
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Classification())
                .learningRate(learningRate)
                .updateFunction(() -> new L2Decay(() -> new ADADELTA(0.9, 0.1), 0.00001f))
                .build();

        trainModel(sgd, reader, epochs, 0);

        return super.getLastValidationResult();
    }

    Model createModel(TensorShape inputs, TensorShape outputs) {
        TensorShape hiddenShape = inputs;
        Model model = new Model(new InputLayer(" In ", inputs, true));
        hiddenShape = addLayer(model, new Convolution2D("Conv 1", hiddenShape, 3, 32, new LRELU()));
        hiddenShape = addLayer(model, new Convolution2D("Conv 2", hiddenShape, 3, 32, new LRELU()));
        hiddenShape = addLayer(model, new PoolMax2D("Pool 1", hiddenShape, 2));
        hiddenShape = addLayer(model, new Convolution2D("Conv 3", hiddenShape, 3, 64, new LRELU()));
        hiddenShape = addLayer(model, new Convolution2D("Conv 4", hiddenShape, 3, 64, new LRELU()));
        hiddenShape = addLayer(model, new PoolMax2D("Pool 2", hiddenShape, 2));
        hiddenShape = addLayer(model, new Flatten("Flatten 1", hiddenShape));
        hiddenShape = addLayer(model, new FullyConnected("FC 1", hiddenShape, 512, new RELU()));
        hiddenShape = addLayer(model, new FullyConnected("FC 2", hiddenShape, 128, new RELU()));
        hiddenShape = addLayer(model, new FullyConnected("FC 3", hiddenShape, 128, new RELU()));
        model.addLayer(new OutputSoftmax("output", hiddenShape, outputs.getNeuronCount(), new CrossEntropy()));
        return model;
    }

    TensorShape addLayer(Model model, Layer new_layer) {
        model.addLayer(new_layer);
        return new_layer.getOutputShape();
    }

    public static void main(String[] args) throws IOException {
        new ExperimentCIFAR10().go(64, 0.1);
    }

    @Override
    public void onEpochFinished(Optimizer sgd, int epoch) {
        super.onEpochFinished(sgd, epoch);
        showCase.update(sgd.getModel());
    }
}

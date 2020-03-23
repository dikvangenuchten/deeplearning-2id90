package experiments;

import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
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
    int epochs = 5;
    double learningRate = 0.1;
    BatchResult localResult;
    ShowCase showCase;

    public BatchResult go(int batchSize, double learningRate) throws IOException {
        // you are going to add code here
        this.batchSize = batchSize;
        this.learningRate = learningRate;

        // read input and print some information on the data
        InputReader reader = new Cifar10Reader(batchSize);

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
                .updateFunction(MomentumGradientDescent::new)
                .build();

        trainModel(sgd, reader, epochs, 0);

        return super.getLastValidationResult();
    }

    Model createModel(TensorShape inputs, TensorShape outputs) {
        Model model = new Model(new InputLayer(" In ", inputs, true));
        TensorShape hiddenShape = addLayer(model, new Flatten("Flatten 1", inputs));
        model.addLayer(new OutputSoftmax("output", hiddenShape, outputs.getNeuronCount(), new CrossEntropy()));
        return model;
    }

    TensorShape addLayer(Model model, Layer new_layer) {
        model.addLayer(new_layer);
        return new_layer.getOutputShape();
    }

    public static void main(String[] args) throws IOException {
        new ExperimentCIFAR10().go(32, 0.1);
    }

    @Override
    public void onEpochFinished(Optimizer sgd, int epoch) {
        super.onEpochFinished(sgd, epoch);
        showCase.update(sgd.getModel());
    }
}

package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class L2Decay implements UpdateFunction {
    double decay;
    UpdateFunction f;

    public L2Decay(Supplier<UpdateFunction> supplier, double decay) {
        this.decay = decay;
        this.f = supplier.get();
    }

    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     *
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        f.update(value, isBias, learningRate, batchSize, gradient);
        if (!isBias) {
            INDArray l2gradient = value.mul(-decay);
            Nd4j.getBlasWrapper().level1().axpy(value.length(), 1, l2gradient, value);
            // value <-- value + factor * gradient
        }
    }
}

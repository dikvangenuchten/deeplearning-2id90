package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Joep en Dik
 */
public class MomentumGradientDescent implements UpdateFunction {

    INDArray prev_gradient = null;
    double momentumUpdate = 0.5;

    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     *
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        if (prev_gradient == null) {
            prev_gradient = Nd4j.zeros(gradient.shape());
        }
        gradient = gradient.addi(prev_gradient);
        prev_gradient = gradient.dup().mul(momentumUpdate);
        double factor = -(learningRate / batchSize);
        Nd4j.getBlasWrapper().level1().axpy(value.length(), factor, gradient, value);
        // value <-- value + factor * gradient
    }
}

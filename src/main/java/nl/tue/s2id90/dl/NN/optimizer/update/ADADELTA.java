package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class ADADELTA implements UpdateFunction {
    private double decay;
    private double e;
    private INDArray gradientAccumulator;
    private INDArray deltaXAccumulator;


    public ADADELTA(double decay, double e) {
        this.decay = decay;
        this.e = e;
    }

    /**
     * Does a gradient descent step with ADADELTA.
     *
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        if (gradientAccumulator == null) {
            gradientAccumulator = Nd4j.zeros(gradient.shape());
            deltaXAccumulator = Nd4j.zeros(gradient.shape());
        }
        // Accumulate and slide gradient
        gradientAccumulator.muli(decay);
        gradientAccumulator.addi(gradient.mul(gradient).mul(1 - decay));

        // Caclulate Delta x
        INDArray deltaX = gradient.div(RMS(gradientAccumulator)).mul(RMS(deltaXAccumulator)).mul(-1);

        // Accumulate and slide deltaX
        deltaXAccumulator.muli(decay);
        deltaXAccumulator.addi(deltaX.mul(deltaX).mul(1 - decay));

        // Update weights
        double factor = 0.1 / batchSize;
        Nd4j.getBlasWrapper().level1().axpy(value.length(), factor, deltaX, value);
        // value <-- value + factor * gradient
    }

    private INDArray RMS(INDArray x) {
        return sqrt(x.add(e));
    }
}

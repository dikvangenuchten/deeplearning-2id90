package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class ADADELTA implements UpdateFunction {
    private double decay = 0.90;
    private double e = 1;
    private int counter = 0;
    private INDArray gradientAccumulator;
    private INDArray deltaXAccumulator;

    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
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
        System.out.println(counter++);
        gradientAccumulator = accumulate(gradientAccumulator, gradient);
        System.out.println("Delta X acc" + RMS(deltaXAccumulator));
        System.out.println("gradient acc" + RMS(gradientAccumulator));
        INDArray RMSdiv = RMS(deltaXAccumulator).div(RMS(gradientAccumulator));
        INDArray deltaX = RMSdiv.mul(gradient).mul(-1);
        deltaXAccumulator = accumulate(deltaXAccumulator, deltaX);
        System.out.println(deltaX);
        Nd4j.getBlasWrapper().level1().axpy(value.length(), 1, deltaX, value);
        // value <-- value + factor * gradient
    }

    private INDArray RMS(INDArray x){
        return sqrt(x.add(this.e));
    }

    private INDArray accumulate(INDArray accumulator, INDArray value){
        INDArray sqrValue = value.mul(value);
        return accumulator.mul(decay).add(sqrValue.mul(1 - decay));
    }
}

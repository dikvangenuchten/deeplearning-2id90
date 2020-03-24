package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class ADAGRAD implements UpdateFunction {
    private double e = 0.1;
    private double decay = 0.90;
    private int counter = 0;
    private INDArray gradientAccumulator;
    private INDArray deltaX;

    /**
     * Does a gradient descent step with ADAGRAD.
     *
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        if(gradientAccumulator == null){
            gradientAccumulator = Nd4j.zeros(gradient.shape());
        }
        gradientAccumulator.muli(decay);
        gradientAccumulator.addi(gradient.mul(gradient).mul(1-decay));
        double factor = -(0.1/batchSize);
        deltaX = gradient.div(RMS(gradientAccumulator));

        Nd4j.getBlasWrapper().level1().axpy( value.length(), factor, deltaX, value );
        // value <-- value + factor * gradient
    }

    private INDArray RMS(INDArray x){
        return sqrt(x.add(this.e));
    }


}

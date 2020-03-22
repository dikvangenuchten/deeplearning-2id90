package experiments;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class MeanSubtraction implements DataTransform {
    double mean = 0;

    @Override
    public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for (TensorPair pair : data) {
            mean += pair.model_input.getValues().meanNumber().doubleValue();
        }
        mean /= data.size();
    }

    @Override
    public void transform(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for (int i = 0; i < data.size(); i++) {
            INDArray values = data.get(i).model_input.getValues();
            Tensor new_input = new Tensor(values.subi(mean), data.get(i).model_input.getShape());
            data.set(i, new TensorPair(new_input, data.get(i).model_output));
        }
    }
}

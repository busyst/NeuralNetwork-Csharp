using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Statistics;
class Net
{
    public Normal normal = new Normal(new Random(123123));
    public List<ILayer> layers = new List<ILayer>();
    public Vector<float> Forward(Vector<float> input)
    {
        var data = input;
        for (int i = 0; i < layers.Count; i++)
            data = layers[i].Forward(data);
        return data;
    }

    public float Backpropagate(Vector<float> X, Vector<float> Y, float learningRate)
    {
        // Calculate the error
        Vector<float> output = Forward(X);
        Vector<float> error = Y-output;
        float p2err = error.PointwisePower(2).Sum(); // Calculate squared error

        for (int i = layers.Count - 1; i >= 0; i--)
        {
            error = layers[i].Backward(error);
        }

        // Update weights and biases
        Parallel.ForEach(layers,(layer)=>
        {
            layer.UpdateWeights(learningRate);
        });

        return p2err;
    }
    public void Compile()
    {
        for (int i = 0; i < layers.Count; i++)
        {
            if (i != layers.Count - 1)
                layers[i].Init(layers[i + 1].NeuronCount, normal);
            else
                layers[i].Init(layers[i].NeuronCount, normal);
        }
    }
}
class DenseLayer : ILayer
{
    public IActivationFunction activationFunction = new None();
    readonly Vector<float> C_input;
    private Matrix<float> weights;
    private Vector<float> bias;

    private Vector<float> C_error;


    public int NeuronCount;

    public DenseLayer(int NeuronCount)
    {
        C_input = Vector<float>.Build.Dense(NeuronCount);
        this.NeuronCount = NeuronCount;
    }

    public void Init(int outputCount, Normal normal)
    {
        weights = Matrix<float>.Build.Random(NeuronCount, outputCount, normal) / (float)(Math.Sqrt(NeuronCount + outputCount));
        bias = Vector<float>.Build.Random(outputCount, normal) / (float)Math.Sqrt(NeuronCount);
        C_error = Vector<float>.Build.Dense(outputCount);
    }
    public Vector<float> Forward(Vector<float> input)
    {
        input.CopyTo(C_input);
        outs = ((input * weights) + bias).Map(activationFunction.Func,Zeros.Include);

        return outs;
    }
    Vector<float> outs;

    int ILayer.NeuronCount => NeuronCount;

    public Vector<float> Backward(Vector<float> error)
    {
        Vector<float> FGradient = outs.Map(activationFunction.DerFunc,Zeros.Include);
        error = error.PointwiseMultiply(FGradient);
        
        error.CopyTo(this.C_error);
        
        return (error * weights.Transpose()).PointwiseMultiply(C_input);
    }

    public void UpdateWeights(float learningRate)
    {
        weights += learningRate * (C_input.ToColumnMatrix() * C_error.ToRowMatrix());
        bias += learningRate * C_error;
    }
}
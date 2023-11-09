using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
public interface IActivationFunction
{
    public float Func(float x);
    public float DerFunc(float x);
}
public interface ILayer
{
    public void Init(int outputCount, Normal normal);
    public Vector<float> Forward(Vector<float> input);
    public Vector<float> Backward(Vector<float> error);
    public void UpdateWeights(float learningRate);
    public int NeuronCount{get;}
}
class None : IActivationFunction
{
    public float DerFunc(float x) =>1;
    public float Func(float x) =>x;
}
class Sigmoid : IActivationFunction
{
    public float Func(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }

    public float DerFunc(float x)
    {
        float sigmoid = Func(x);
        return sigmoid * (1.0f - sigmoid);
    }
}

class Tanh : IActivationFunction
{
    public float Func(float x)
    {
        return MathF.Tanh(x);
    }

    public float DerFunc(float x)
    {
        float tanh = Func(x);
        return 1.0f - tanh * tanh;
    }
}
class ReLu : IActivationFunction
{
    public float Func(float x)
    {
        return MathF.Max(x,0);
    }

    public float DerFunc(float x)=>x>0?1f:0f;
}
class LeakyReLu : IActivationFunction
{
    public float steep = 0.2f;
    public float Func(float x)
    {
        return x>0?x:steep*x;
    }

    public float DerFunc(float x)=>x>0?1f:steep;
}
class ELu : IActivationFunction
{
    public float Func(float x)
    {
        return x>0?x:MathF.Exp(x)-1;
    }

    public float DerFunc(float x)=>x>0?1f:MathF.Exp(x);
}
class GeLu : IActivationFunction
{
    public float Func(float x)
    {
        return x>0?x:MathF.Exp(-(x*x))*x;
    }

    public float DerFunc(float x)=>x>0?1f:(MathF.Exp(-(x*x))*(1-(2*x*x)));
}
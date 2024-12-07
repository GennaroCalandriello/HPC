__device__ __inline__ 
void complexProd(cuDoubleComplex* a, double b)
//x, y risp. parte reale e immaginaria come definite nella libreria cuComplex.h
{
    a -> x *= b;
    a -> y *= b;
}

__device__ __inline__ cuDoubleComplex exp_complex(cuDoubleComplex a)
{
    cuDoubleComplex res;
    res.x = exp(a.x)*cos(a.y);
    res.y = exp(a.x)*sin(a.y);
    return res;
}

__device__ __inline__ void complexDiv(cuDoubleComplex* n, double d)
{
    n -> x /= d;
    n -> y /= d;
}
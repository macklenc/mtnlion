class Composition : public Expression
{
public:
    Composition() : Expression() {}

    void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const
    {
        Array<double> val(3);
        inner->eval(val, x, c);
        outer->eval(values, val, c);      
    }

    std::shared_ptr<GenericFunction> outer;
    std::shared_ptr<GenericFunction> inner;
};
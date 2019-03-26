class Composition : public Expression
{
public:
    Composition() : Expression() {}

    void eval(Eigen::Ref<Eigen::VectorXd>& values, Eigen::Ref<const Eigen::VectorXd>& x, const ufc::cell& c) const
    {
        Array<double> val(3);
        inner->eval(val, x, c);
        outer->eval(values, val, c);      
    }

    std::shared_ptr<GenericFunction> outer;
    std::shared_ptr<GenericFunction> inner;
};

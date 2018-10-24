class Piecewise : public Expression
{
    public:
        Piecewise() : Expression() {}

        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
        {
            switch((*markers)[cell.index]){
            case 0:
                k_1->eval(values, x);
                break;
            case 1:
                k_2->eval(values, x);
                break;
            case 2:
                k_3->eval(values, x);
                break;
            case 3:
                k_4->eval(values, x);
                break;
            }
        }

    std::shared_ptr<MeshFunction<std::size_t>> markers;
    std::shared_ptr<GenericFunction> k_1;
    std::shared_ptr<GenericFunction> k_2;
    std::shared_ptr<GenericFunction> k_3;
    std::shared_ptr<GenericFunction> k_4;
};
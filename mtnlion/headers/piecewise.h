class Piecewise : public Expression
{
    public:
        Piecewise : Expression() {}
        
        void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
        {
            switch((*markers)[cell.index]){
            case 1:
                values[0] = k_1;
                break;
            case 2:
                values[0] = k_2;
                break;
            case 3:
                values[0] = k_3;
                break;
            case 4:
                values[0] = k_4;
                break;
            default:
                values[0] = 0;
            }
        }

    std::shared_ptr<MeshFunction<std::size_t>> markers;
    double k_1, k_2, k_3, k_4;
};
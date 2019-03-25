class XBar : public Expression
{
public:
	XBar() : Expression() {}

	void eval(Eigen::Ref<Eigen::VectorXd>& values, Eigen::Ref<const Eigen::VectorXd>& x, const ufc::cell& c) const {
		switch((*markers)[c.index]){
			case 0:
				neg->eval(values, x, c);
				break;
			case 1:
				sep->eval(values, x, c);
				break;
			case 2:
				pos->eval(values, x, c);
				break;
		}
	}

	std::shared_ptr<MeshFunction<std::size_t>> markers;
	std::shared_ptr<GenericFunction> neg;
	std::shared_ptr<GenericFunction> sep;
	std::shared_ptr<GenericFunction> pos;
};

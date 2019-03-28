class XBarSimple : public Expression
{
public:
	XBarSimple() : Expression() {}

	void eval(Eigen::Ref<Eigen::VectorXd>& values, Eigen::Ref<const Eigen::VectorXd>& x, const ufc::cell& c) const {
	    xbar->eval(values, x, c);
	}

	std::shared_ptr<GenericFunction> xbar;
};
